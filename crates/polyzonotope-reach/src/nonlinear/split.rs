//! Splitting a set that is too large for one step, and merging pieces back
//! when that costs little.
//!
//! A split cuts the domain of one factor in half
//! ([`SparsePolyZonotope::split`], exact), either of the piece itself, which
//! keeps its dependencies, or of its oriented box, whichever leaves the
//! smaller halves. A merge encloses the convex hull of two pieces with a
//! polynomial zonotope over their factors, paired up so that two halves of
//! a factor split merge back into their parent.

use alloc::vec::Vec;

use super::Splitting;
use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, Interval, IntervalBox, Reduction, SparsePolyZonotope, Zonotope};

/// `piece` in two halves whose larger `cost` is as small as one split can
/// make it, or `None` if no split lowers the cost.
///
/// The caller's cost is the size that makes a step fail: for
/// [`NonlinearReach`](super::NonlinearReach), the second-order error bound
/// of the piece. The width alone would pick the wrong factor: a set wide
/// along a direction where `f` is nearly linear does not need splitting
/// there.
///
/// Two families of splits compete:
///
/// - Factor splits keep every dependency. After reduction most of a set's
///   size sits in its independent generators, which no factor split can
///   touch, so they are first enclosed in their oriented box
///   ([`Reduction::Pca`] of the independent part alone, which loses no
///   dependency, since they have none), and the box's `n` generators
///   become dependent factors of their own. That is exact: `g β` with
///   `β ∈ [-1, 1]` is the monomial `g α` of a factor that appears nowhere
///   else. Without the box, a split would cut one of dozens of small
///   generators and barely shrink the set.
/// - Box splits enclose the whole piece in its oriented box first and cut
///   that along one axis. They lose the dependencies but halve the extent
///   along the axis, where a factor split halves one generator's share of
///   it.
///
/// On the cubic drift `x₀' = x₀³` from a box of radius 0.5, factor splits
/// alone needed 30 pieces where box splits needed fewer than 8, so the
/// cheaper of the two wins, split by split, unless `splitting` is
/// [`Splitting::Factors`].
pub(super) fn halves(
    piece: &SparsePolyZonotope,
    cost: impl Fn(&SparsePolyZonotope) -> f64,
    splitting: Splitting,
) -> Option<(SparsePolyZonotope, SparsePolyZonotope)> {
    let lifted = lift_independent(&box_independent(piece));
    let oriented = match splitting {
        Splitting::Factors => None,
        _ => Some(SparsePolyZonotope::from_zonotope(
            &piece.reduce(1.0, Reduction::Pca).to_zonotope(),
        )),
    };
    let candidates = core::iter::once(&lifted)
        .chain(oriented.as_ref())
        .flat_map(|set| set.ids().iter().map(move |&id| (set, id)))
        .filter(|&(set, id)| splittable(set, id));
    let best = candidates
        .map(|(set, id)| {
            let (lower, upper) = set.split(id);
            (cost(&lower).max(cost(&upper)), lower, upper)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))?;

    (best.0 < cost(piece)).then_some((best.1, best.2))
}

/// `pz` with its independent generators replaced by their oriented box:
/// an enclosure that keeps every dependent monomial as it is.
fn box_independent(pz: &SparsePolyZonotope) -> SparsePolyZonotope {
    let n = pz.dim();
    let independent = Zonotope::new(DVector::zeros(n), pz.independent_generators().clone())
        .reduce(1.0, Reduction::Pca);

    SparsePolyZonotope::new(
        pz.center().clone(),
        pz.dependent_generators().clone(),
        independent.generators().clone(),
        pz.exponents().clone(),
        pz.ids().to_vec(),
    )
    .expect("same shapes")
    .translate(independent.center())
}

/// `pz` with every independent generator turned into a dependent factor of
/// its own; the same set.
pub(super) fn lift_independent(pz: &SparsePolyZonotope) -> SparsePolyZonotope {
    let (dependent, independent, powers) = (
        pz.dependent_generators(),
        pz.independent_generators(),
        pz.exponents(),
    );
    let (deps, lifts, factors) = (dependent.ncols(), independent.ncols(), pz.ids().len());
    let generators = DMatrix::from_fn(pz.dim(), deps + lifts, |i, j| {
        if j < deps {
            dependent[(i, j)]
        } else {
            independent[(i, j - deps)]
        }
    });
    let exponents = DMatrix::from_fn(factors + lifts, deps + lifts, |k, j| {
        match (k < factors, j < deps) {
            (true, true) => powers[(k, j)],
            (false, false) => u32::from(k - factors == j - deps),
            _ => 0,
        }
    });
    let mut ids = pz.ids().to_vec();
    ids.extend(FactorId::fresh_n(lifts));

    SparsePolyZonotope::new(
        pz.center().clone(),
        generators,
        DMatrix::zeros(pz.dim(), 0),
        exponents,
        ids,
    )
    .expect("lifted shapes fit")
}

/// Whether [`SparsePolyZonotope::split`] can take `factor`: its exponents
/// stay at most 56, where the binomial coefficients are exact.
fn splittable(pz: &SparsePolyZonotope, factor: FactorId) -> bool {
    let k = pz
        .ids()
        .iter()
        .position(|&id| id == factor)
        .expect("factor of this set");

    pz.exponents().row(k).iter().all(|&e| e <= 56)
}

/// The sum of the widths of the interval hull: the size measure of both
/// splitting and merging.
pub(super) fn width(pz: &SparsePolyZonotope) -> f64 {
    pz.interval_hull().0.iter().map(Interval::width).sum()
}

/// Merges pairs of `pieces` while some pair's merged enclosure has a hull
/// of at most `1 + tolerance` times the volume of the two pieces' hulls
/// together, and is no wider than `cap`. The pair with the smallest ratio
/// goes first. Volume, not width: two far-apart boxes span a box barely
/// wider than the two, but mostly empty.
///
/// `cap` keeps a merge from undoing a split: a piece is split because it is
/// too large to step, so the caller passes a width below the pieces that
/// failed.
/// `finish` reduces each merged piece, since a merge adds generators.
pub(super) fn merge(
    mut pieces: Vec<SparsePolyZonotope>,
    tolerance: f64,
    cap: f64,
    finish: impl Fn(&SparsePolyZonotope) -> SparsePolyZonotope,
) -> Vec<SparsePolyZonotope> {
    while let Some((i, j, merged)) = best_merge(&pieces, tolerance, cap) {
        pieces.swap_remove(j);
        pieces[i] = finish(&merged);
    }

    pieces
}

/// The pair `(i, j)`, `i < j`, of `pieces` whose merge is cheapest within
/// `tolerance` and `cap`, with that merge.
fn best_merge(
    pieces: &[SparsePolyZonotope],
    tolerance: f64,
    cap: f64,
) -> Option<(usize, usize, SparsePolyZonotope)> {
    let hulls: Vec<_> = pieces
        .iter()
        .map(SparsePolyZonotope::interval_hull)
        .collect();
    let mut best: Option<(f64, usize, usize, SparsePolyZonotope)> = None;
    for j in 1..pieces.len() {
        for i in 0..j {
            let merged = hull_of_pair(&pieces[i], &pieces[j]);
            let ratio = volume(&merged.interval_hull()) / (volume(&hulls[i]) + volume(&hulls[j]));
            let fits = width(&merged) <= cap && ratio <= 1.0 + tolerance;
            if fits && best.as_ref().is_none_or(|b| ratio < b.0) {
                best = Some((ratio, i, j, merged));
            }
        }
    }

    best.map(|(_, i, j, merged)| (i, j, merged))
}

/// The volume of `hull`, each side at least `1e-12` so that a flat box
/// still compares.
fn volume(hull: &IntervalBox) -> f64 {
    hull.0.iter().map(|iv| iv.width().max(1e-12)).product()
}

/// An enclosure of the convex hull of `a ∪ b` that correlates their
/// factors:
///
/// `λ a + (1 − λ) b = ½ (a + b) + ν ½ (a − b)`, `ν = 2λ − 1 ∈ [-1, 1]`.
///
/// For a point of `a`, any point of `b` will do at `ν = 1`, so `b` may be
/// evaluated at whatever factor values suit: every value in `[-1, 1]` is
/// allowed for every factor. Shared factors keep one value on both sides,
/// and [`pair_factors`] gives the factors only one side has the values of
/// the other side's. For two halves of a split, that turns the merge back
/// into the parent. Both halves are added with
/// [`SparsePolyZonotope::exact_plus`], and `ν` becomes a fresh dependent
/// factor that multiplies every monomial of `½ (a − b)`. At `ν = 1` the
/// independent generators of `b` cancel when both sides use the same
/// coefficients, and those of `a` add up to their full size; the same at
/// `ν = −1` for `b`. So both pieces lie in the result.
fn hull_of_pair(a: &SparsePolyZonotope, b: &SparsePolyZonotope) -> SparsePolyZonotope {
    let b = pair_factors(a, b);
    let n = a.dim();
    let half = DMatrix::from_diagonal_element(n, n, 0.5);
    let mean = a.linear_map(&half).exact_plus(&b.linear_map(&half));
    let difference = a.linear_map(&half).exact_plus(&b.linear_map(&-&half));

    mean.exact_plus(&scaled_by_fresh_factor(&difference))
}

/// `b` with the factors it does not share with `a` renamed, in order, to
/// those of `a` that `b` does not have, as far as both lists go. The same
/// set; only its correlation with `a` changes.
fn pair_factors(a: &SparsePolyZonotope, b: &SparsePolyZonotope) -> SparsePolyZonotope {
    let mut only_a = a.ids().iter().filter(|id| !b.ids().contains(id));
    let ids = b
        .ids()
        .iter()
        .map(|id| {
            if a.ids().contains(id) {
                *id
            } else {
                only_a.next().copied().unwrap_or(*id)
            }
        })
        .collect();

    SparsePolyZonotope::new(
        b.center().clone(),
        b.dependent_generators().clone(),
        b.independent_generators().clone(),
        b.exponents().clone(),
        ids,
    )
    .expect("renamed ids stay distinct")
}

/// `{ ν p | p ∈ pz, ν ∈ [-1, 1] }` with `ν` a fresh dependent factor: the
/// center and every dependent monomial gain a power of `ν`, and the
/// independent generators stay, since `ν β` again ranges over `[-1, 1]`.
fn scaled_by_fresh_factor(pz: &SparsePolyZonotope) -> SparsePolyZonotope {
    let (dependent, powers) = (pz.dependent_generators(), pz.exponents());
    let (deps, factors) = (dependent.ncols(), pz.ids().len());
    let generators = DMatrix::from_fn(pz.dim(), deps + 1, |i, j| {
        if j < deps {
            dependent[(i, j)]
        } else {
            pz.center()[i]
        }
    });
    let exponents = DMatrix::from_fn(factors + 1, deps + 1, |k, j| {
        match (k < factors, j < deps) {
            (true, true) => powers[(k, j)],
            (true, false) => 0,
            (false, _) => 1,
        }
    });
    let mut ids = pz.ids().to_vec();
    ids.push(FactorId::fresh());

    SparsePolyZonotope::new(
        DVector::zeros(pz.dim()),
        generators,
        pz.independent_generators().clone(),
        exponents,
        ids,
    )
    .expect("scaled shapes fit")
    .compact()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box with one dependent and one independent generator.
    fn mixed() -> SparsePolyZonotope {
        SparsePolyZonotope::new(
            DVector::from_vec(alloc::vec![1.0, 2.0]),
            DMatrix::from_row_slice(2, 1, &[1.0, 0.5]),
            DMatrix::from_row_slice(2, 1, &[0.0, 3.0]),
            DMatrix::from_element(1, 1, 1),
            FactorId::fresh_n(1),
        )
        .unwrap()
    }

    /// Lifting keeps the set: evaluating with the new factor at `β` gives
    /// the point of the original at `β`.
    #[test]
    fn lifting_is_exact() {
        let pz = mixed();
        let lifted = lift_independent(&pz);
        assert_eq!(lifted.independent_generators().ncols(), 0);
        for (a, b) in [(-1.0, 0.5), (0.3, -0.7), (1.0, 1.0)] {
            let gap = lifted.evaluate(&[a, b], &[]) - pz.evaluate(&[a], &[b]);
            assert!(gap.amax() < 1e-12, "{gap}");
        }
    }

    /// The split goes along the independent generator, the widest part, and
    /// the halves are narrower than the set.
    #[test]
    fn split_takes_the_widest_factor() {
        let pz = mixed();
        let (lower, upper) = halves(&pz, width, Splitting::Cheapest).expect("splittable");
        assert!(width(&lower) < width(&pz) && width(&upper) < width(&pz));
        let hull = lower.interval_hull();
        assert!((hull.0[1].width() - 4.0).abs() < 1e-12, "{hull:?}");
    }

    /// A point cannot be split.
    #[test]
    fn a_point_has_no_halves() {
        let point = SparsePolyZonotope::from_zonotope(&Zonotope::new(
            DVector::from_element(2, 1.0),
            DMatrix::zeros(2, 0),
        ));
        assert!(halves(&point, width, Splitting::Cheapest).is_none());
    }

    /// Two halves of a split merge back into a set no wider than their
    /// parent, so the merge fires.
    #[test]
    fn halves_merge_back() {
        let pz = mixed();
        let (lower, upper) = halves(&pz, width, Splitting::Factors).unwrap();
        let merged = merge(
            alloc::vec![lower.clone(), upper.clone()],
            0.05,
            f64::INFINITY,
            Clone::clone,
        );
        assert_eq!(merged.len(), 1);
        assert!(width(&merged[0]) <= width(&pz) + 1e-12);
    }

    /// The cap stops a merge into a set wider than the caller can step.
    #[test]
    fn cap_blocks_the_merge() {
        let (lower, upper) = halves(&mixed(), width, Splitting::Factors).unwrap();
        let cap = width(&lower).max(width(&upper));
        assert_eq!(
            merge(alloc::vec![lower, upper], 0.05, cap, Clone::clone).len(),
            2
        );
    }

    /// Far-apart pieces stay separate: their hull is mostly empty space.
    #[test]
    fn distant_pieces_stay_apart() {
        let at = |x: f64| {
            SparsePolyZonotope::from_zonotope(&Zonotope::new(
                DVector::from_vec(alloc::vec![x, 0.0]),
                DMatrix::from_diagonal_element(2, 2, 0.1),
            ))
        };
        assert_eq!(
            merge(
                alloc::vec![at(0.0), at(5.0)],
                0.1,
                f64::INFINITY,
                Clone::clone
            )
            .len(),
            2
        );
    }
}

#[cfg(test)]
mod properties {
    use super::*;
    use proptest::prelude::*;

    /// A random 2-D set with two dependent factors of degree up to 2 and one
    /// independent generator.
    fn arb_set() -> impl Strategy<Value = SparsePolyZonotope> {
        (
            proptest::collection::vec(-2.0..2.0f64, 2),
            proptest::collection::vec(-1.0..1.0f64, 6),
            proptest::collection::vec(0u32..3, 6),
            proptest::collection::vec(-0.5..0.5f64, 2),
        )
            .prop_map(|(c, g, e, gi)| {
                SparsePolyZonotope::new(
                    DVector::from_vec(c),
                    DMatrix::from_vec(2, 3, g),
                    DMatrix::from_vec(2, 1, gi),
                    DMatrix::from_vec(2, 3, e),
                    FactorId::fresh_n(2),
                )
                .unwrap()
            })
    }

    /// Whether `p` lies in the interval hull of `pz` and of every piece of
    /// a few splits of it: a cheap but sharp necessary test for membership.
    fn near(pz: &SparsePolyZonotope, p: &DVector<f64>) -> bool {
        pz.split_pieces(6)
            .iter()
            .any(|piece| piece.interval_hull().contains(p, 1e-9))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The halves of a split cover the set: each sampled point of the
        /// set is in one of them, at the same factor values.
        #[test]
        fn halves_cover_the_set(pz in arb_set(), a in -1.0..1.0f64, b in -1.0..1.0f64, beta in -1.0..1.0f64) {
            let p = pz.evaluate(&[a, b], &[beta]);
            if let Some((lower, upper)) = halves(&pz, width, Splitting::Cheapest) {
                prop_assert!(near(&lower, &p) || near(&upper, &p));
            }
        }

        /// A merge holds sampled points of both pieces, also of pieces that
        /// share factors and of unrelated ones.
        #[test]
        fn merge_holds_both_pieces(
            x in arb_set(),
            y in arb_set(),
            shared in proptest::bool::ANY,
            a in -1.0..1.0f64, b in -1.0..1.0f64, beta in -1.0..1.0f64,
        ) {
            let y = if shared { pair_factors(&x, &y) } else { y };
            let merged = hull_of_pair(&x, &y);
            prop_assert!(near(&merged, &x.evaluate(&[a, b], &[beta])));
            prop_assert!(near(&merged, &y.evaluate(&[a, b], &[beta])));
        }
    }
}

//! Soundness and exactness properties, checked on random sets and random samples.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{
    Containment, FactorId, Halfspace, Intersection, Interval, IntervalBox, Polytope, Reduction,
    SparsePolyZonotope, Zonotope,
};
use proptest::collection::vec;
use proptest::prelude::*;

use crate::strategies::{Sample, TOL, alpha_map, arb_pz, arb_samples, close, eval, pad};

proptest! {
    /// Every sampled point of a set lies in its interval hull.
    #[test]
    fn interval_hull_contains_every_point(pz in arb_pz(), samples in arb_samples()) {
        let hull = pz.interval_hull();
        for s in &samples {
            prop_assert!(hull.contains(&eval(&pz, s), TOL));
        }
    }

    /// Soundness *by construction*: for every point of the PZ we exhibit the
    /// zonotope coefficients that reproduce it, and check they lie in [-1, 1].
    #[test]
    fn to_zonotope_reproduces_every_point(pz in arb_pz(), samples in arb_samples()) {
        let z = pz.to_zonotope();
        for s in &samples {
            let coeffs = zonotope_coefficients(&pz, s);

            prop_assert!(coeffs.iter().all(|c| c.abs() <= 1.0 + TOL));
            prop_assert!(close(&z.evaluate(&pad(&coeffs, z.num_generators())), &eval(&pz, s)));
        }
    }

    /// The zonotope enclosure loses nothing on the box level: its interval
    /// hull equals the polynomial zonotope's.
    #[test]
    fn to_zonotope_keeps_the_interval_hull(pz in arb_pz()) {
        let direct = pz.interval_hull();
        let via_zonotope = pz.to_zonotope().interval_hull();
        for (a, b) in direct.0.iter().zip(&via_zonotope.0) {
            prop_assert!((a.lo - b.lo).abs() <= TOL * (1.0 + a.lo.abs()));
            prop_assert!((a.hi - b.hi).abs() <= TOL * (1.0 + a.hi.abs()));
        }
    }

    /// Every sampled point of a zonotope lies in its interval hull.
    #[test]
    fn zonotope_hull_contains_every_point(pz in arb_pz(), samples in arb_samples()) {
        let z = pz.to_zonotope();
        let hull = z.interval_hull();
        for s in &samples {
            let beta: Vec<f64> = (0..z.num_generators()).map(|j| s.beta[j % s.beta.len()]).collect();
            prop_assert!(hull.contains(&z.evaluate(&beta), TOL));
        }
    }

    /// A box survives the round trip through [`Zonotope::from_box`] unchanged.
    #[test]
    fn from_box_round_trips_the_hull(pz in arb_pz()) {
        let hull = pz.interval_hull();
        let round_trip = Zonotope::from_box(&hull).interval_hull();
        for (a, b) in hull.0.iter().zip(&round_trip.0) {
            prop_assert!((a.lo - b.lo).abs() <= TOL * (1.0 + a.lo.abs()));
            prop_assert!((a.hi - b.hi).abs() <= TOL * (1.0 + a.hi.abs()));
        }
    }

    /// Linear maps commute with evaluation: `(M PZ)(α, β) = M PZ(α, β)`.
    #[test]
    fn linear_map_is_exact(pz in arb_pz(), m in vec(-3.0..3.0f64, 9), samples in arb_samples()) {
        let n = pz.dim();
        let m = DMatrix::from_fn(n, n, |i, j| m[i * 3 + j]);
        let mapped = pz.linear_map(&m);
        for s in &samples {
            prop_assert!(close(&eval(&mapped, s), &(&m * eval(&pz, s))));
        }
    }

    /// Translation commutes with evaluation: `(PZ + v)(α, β) = PZ(α, β) + v`.
    #[test]
    fn translate_is_exact(pz in arb_pz(), v in vec(-5.0..5.0f64, 3), samples in arb_samples()) {
        let v = DVector::from_column_slice(&v[..pz.dim()]);
        let moved = pz.translate(&v);
        prop_assert!(close(moved.center(), &(pz.center() + &v)));
        for s in &samples {
            prop_assert!(close(&eval(&moved, s), &(eval(&pz, s) + &v)));
        }
    }

    /// The exact zonotope operations commute with evaluation: linear maps and
    /// translations act on each point, and a Minkowski sum adds the points
    /// of both operands for their own coefficients.
    #[test]
    fn zonotope_operations_are_exact(
        pz in arb_pz(),
        m in vec(-3.0..3.0f64, 9),
        v in vec(-5.0..5.0f64, 3),
        samples in arb_samples(),
    ) {
        let z = pz.to_zonotope();
        let n = z.dim();
        let m = DMatrix::from_fn(n, n, |i, j| m[i * 3 + j]);
        let v = DVector::from_column_slice(&v[..n]);
        let (mapped, moved, sum) = (z.linear_map(&m), z.translate(&v), z.minkowski_sum(&z));
        let k = z.num_generators();
        for s in &samples {
            let coeffs: Vec<f64> = (0..k).map(|j| s.beta[j % s.beta.len()]).collect();
            let other: Vec<f64> = (0..k).map(|j| s.alpha[j % s.alpha.len()]).collect();
            let both: Vec<f64> = coeffs.iter().chain(&other).copied().collect();
            let point = z.evaluate(&coeffs);

            let at = |z: &Zonotope, coeffs: &[f64]| z.evaluate(&pad(coeffs, z.num_generators()));
            prop_assert!(close(&at(&mapped, &coeffs), &(&m * &point)));
            prop_assert!(close(&at(&moved, &coeffs), &(&point + &v)));
            prop_assert!(close(&at(&sum, &both), &(&point + z.evaluate(&other))));
        }
    }

    /// Converting a zonotope turns each generator into its own factor, so
    /// the coefficients of the zonotope serve as the α values unchanged.
    #[test]
    fn from_zonotope_is_exact(pz in arb_pz(), samples in arb_samples()) {
        let z = pz.to_zonotope();
        let converted = SparsePolyZonotope::from_zonotope(&z);
        for s in &samples {
            // Cyclic, so the coefficient count is not bounded by the sample length.
            let coeffs: Vec<f64> = (0..z.num_generators()).map(|j| s.beta[j % s.beta.len()]).collect();
            prop_assert!(close(&converted.evaluate(&coeffs, &[]), &z.evaluate(&coeffs)));
        }
    }

    /// Positional evaluation agrees with evaluation by id.
    #[test]
    fn evaluate_matches_evaluate_with(pz in arb_pz(), samples in arb_samples()) {
        let (p, q) = (pz.ids().len(), pz.independent_generators().ncols());
        for s in &samples {
            prop_assert!(close(&pz.evaluate(&s.alpha[..p], &s.beta[..q]), &eval(&pz, s)));
        }
    }

    /// Compacting changes the representation, not the function of α and β.
    #[test]
    fn compact_preserves_the_function(pz in arb_pz(), samples in arb_samples()) {
        let compacted = pz.compact();
        for s in &samples {
            // compact() keeps the relative order of ids, so look values up by id.
            let a = alpha_map(pz.ids(), s);
            let q = pz.independent_generators().ncols();
            let beta = &s.beta[..q];

            // The original generators keep their positions; `sound` may append error bounds after them.
            let q_compacted = compacted.independent_generators().ncols();
            // Plain mode adds nothing; sound mode one error generator per dimension at most.
            let growth = if cfg!(feature = "sound") { pz.dim() } else { 0 };
            prop_assert!(q <= q_compacted && q_compacted <= q + growth);
            prop_assert_eq!(compacted.independent_generators().columns(0, q).into_owned(), pz.independent_generators().clone());
            prop_assert!(close(
                &compacted.evaluate_with(|k| a[&k], &pad(beta, q_compacted)),
                &pz.evaluate_with(|k| a[&k], beta),
            ));
        }
    }

    /// Shared factors stay correlated: (p + Mp)(α) = p(α) + M p(α).
    #[test]
    fn exact_plus_respects_shared_factors(pz in arb_pz(), samples in arb_samples()) {
        let n = pz.dim();
        let m = DMatrix::from_fn(n, n, |i, j| if i == j { 2.0 } else { 0.5 });
        let other = pz.linear_map(&m);
        let sum = pz.exact_plus(&other);
        let q = pz.independent_generators().ncols();
        for s in &samples {
            let a = alpha_map(pz.ids(), s);
            let beta = &s.beta[..q];
            let expected = pz.evaluate_with(|k| a[&k], beta) + other.evaluate_with(|k| a[&k], &pad(beta, other.independent_generators().ncols()));

            // Independent generators are concatenated, so feed the same β twice.
            let twice: Vec<f64> = beta.iter().chain(beta).copied().collect();
            let twice = pad(&twice, sum.independent_generators().ncols());
            prop_assert!(close(&sum.evaluate_with(|k| a[&k], &twice), &expected));
        }
    }

    /// The Minkowski sum of a set with itself decorrelates the two copies:
    /// `(p ⊕ p)(α₁, α₂) = p(α₁) + p(α₂)` for independent `α₁`, `α₂`.
    #[test]
    fn minkowski_sum_decorrelates_the_operands(pz in arb_pz(), s1 in arb_samples(), s2 in arb_samples()) {
        let sum = pz.minkowski_sum(&pz);
        // The second copy got fresh ids; compaction keeps the same factors of both copies, in order.
        let (old, new): (Vec<FactorId>, Vec<FactorId>) =
            sum.ids().iter().copied().partition(|id| pz.ids().contains(id));
        prop_assert_eq!(old.len(), new.len());

        let q = pz.independent_generators().ncols();
        for (s1, s2) in s1.iter().zip(&s2) {
            let (a1, a2) = (alpha_map(pz.ids(), s1), alpha_map(pz.ids(), s2));
            let value = |id: FactorId| match new.iter().position(|&n| n == id) {
                Some(j) => a2[&old[j]],
                None => a1[&id],
            };
            let beta: Vec<f64> = s1.beta[..q].iter().chain(&s2.beta[..q]).copied().collect();
            let beta = pad(&beta, sum.independent_generators().ncols());
            prop_assert!(close(&sum.evaluate_with(value, &beta), &(eval(&pz, s1) + eval(&pz, s2))));
        }
    }

    /// The Cartesian product of a set with its own image stacks both points
    /// for the same α: `(p × Mp)(α) = (p(α), M p(α))`.
    #[test]
    fn cartesian_product_keeps_shared_factors(pz in arb_pz(), samples in arb_samples()) {
        let n = pz.dim();
        let m = DMatrix::from_fn(n, n, |i, j| if i == j { -1.0 } else { 0.25 });
        let other = pz.linear_map(&m);
        let product = pz.cartesian_product(&other);
        let q = pz.independent_generators().ncols();
        for s in &samples {
            let a = alpha_map(pz.ids(), s);
            let beta = &s.beta[..q];
            let top = pz.evaluate_with(|k| a[&k], beta);
            let bottom = other.evaluate_with(|k| a[&k], &pad(beta, other.independent_generators().ncols()));
            let expected = DVector::from_iterator(2 * n, top.iter().chain(bottom.iter()).copied());

            let twice: Vec<f64> = beta.iter().chain(beta).copied().collect();
            let twice = pad(&twice, product.independent_generators().ncols());
            prop_assert!(close(&product.evaluate_with(|k| a[&k], &twice), &expected));
        }
    }

    /// Each half of a split reproduces the original set on its half of the
    /// factor's domain, via the rescaling `α = (α' ∓ 1) / 2`.
    #[test]
    fn split_halves_are_exact(pz in arb_pz(), samples in arb_samples()) {
        let k = pz.ids()[0];
        let (lo, hi) = pz.split(k);
        for s in &samples {
            let a = alpha_map(pz.ids(), s);
            let t = s.alpha[7]; // the rescaled factor α'
            let q = pz.independent_generators().ncols();
            for (half, sign) in [(&lo, -1.0), (&hi, 1.0)] {
                let new_id = half.ids().iter().copied().find(|id| !pz.ids().contains(id));
                let value = |id: FactorId| if Some(id) == new_id { t } else { a[&id] };
                let orig = |id: FactorId| if id == k { f64::midpoint(t, sign) } else { a[&id] };
                let beta = &s.beta[..q];
                let padded = pad(beta, half.independent_generators().ncols());
                prop_assert!(close(&half.evaluate_with(value, &padded), &pz.evaluate_with(orig, beta)));
            }
        }
    }
}

proptest! {
    /// Every point `x` of a set maps to `(xᵀ Qᵢ x)ᵢ`, which lies in the hull of
    /// the quadratic map; without independent generators the map is exact.
    #[test]
    fn quadratic_map_commutes_with_evaluation(
        pz in arb_pz(),
        q in vec(-2.0..2.0f64, 18),
        outputs in 1usize..=2,
        samples in arb_samples(),
    ) {
        let n = pz.dim();
        let q: Vec<DMatrix<f64>> = (0..outputs)
            .map(|o| DMatrix::from_fn(n, n, |i, j| q[o * 9 + i * 3 + j]))
            .collect();
        let mapped = pz.quadratic_map(&q);
        let hull = mapped.interval_hull();
        let exact = pz.independent_generators().ncols() == 0;
        for s in &samples {
            let x = eval(&pz, s);
            let y = DVector::from_fn(outputs, |o, _| x.dot(&(&q[o] * &x)));

            prop_assert!(hull.contains(&y, TOL * (1.0 + y.amax())));
            if exact {
                let a = alpha_map(pz.ids(), s);
                let beta = pad(&[], mapped.independent_generators().ncols());
                prop_assert!(close(&mapped.evaluate_with(|k| a[&k], &beta), &y));
            }
        }
    }
}

proptest! {
    /// A reduced zonotope contains the original: its support function is at
    /// least as large in every sampled direction, and it meets the order.
    #[test]
    fn zonotope_reduce_encloses_the_original(
        pz in arb_pz(),
        extra in arb_pz(),
        order in 1.0..2.5f64,
        pca in any::<bool>(),
        directions in vec(vec(-1.0..1.0f64, 3), 1..10),
    ) {
        let z = pz.to_zonotope();
        // Stack more generators so the reduction has something to do.
        let z = z.minkowski_sum(&z.linear_map(&DMatrix::from_fn(z.dim(), z.dim(), |i, j| {
            extra.center()[(i + j) % extra.dim()]
        })));
        let method = if pca { Reduction::Pca } else { Reduction::Girard };
        let reduced = z.reduce(order, method);

        prop_assert!(reduced.num_generators() <= max_generators(order, z.dim(), method));
        for d in &directions {
            let d = DVector::from_column_slice(&d[..z.dim()]);
            let (outer, inner) = (support(&reduced, &d), support(&z, &d));
            prop_assert!(outer >= inner - TOL * (1.0 + inner.abs()), "{outer} < {inner}");
        }
    }

    /// Every sampled point of a set lies in the reduced set, as seen by the
    /// support function of the reduced set's zonotope enclosure.
    #[test]
    fn pz_reduce_encloses_every_point(
        pz in arb_pz(),
        order in 1.0..2.5f64,
        pca in any::<bool>(),
        directions in vec(vec(-1.0..1.0f64, 3), 1..10),
        samples in arb_samples(),
    ) {
        let method = if pca { Reduction::Pca } else { Reduction::Girard };
        let reduced = pz.reduce(order, method);
        let count = reduced.dependent_generators().ncols() + reduced.independent_generators().ncols();
        let enclosure = reduced.to_zonotope();

        prop_assert!(count <= max_generators(order, pz.dim(), method));
        for s in &samples {
            let x = eval(&pz, s);
            for d in &directions {
                let d = DVector::from_column_slice(&d[..pz.dim()]);
                let h = support(&enclosure, &d);
                prop_assert!(d.dot(&x) <= h + TOL * (1.0 + h.abs()));
            }
        }
    }
}

proptest! {
    /// The split hull contains every sampled point and lies inside the plain hull.
    #[test]
    fn split_hull_is_sound_and_tighter(pz in arb_pz(), splits in 0usize..20, samples in arb_samples()) {
        let (tight, coarse) = (pz.interval_hull_split(1e-3, splits), pz.interval_hull());
        for s in &samples {
            prop_assert!(tight.contains(&eval(&pz, s), TOL));
        }
        for (t, c) in tight.0.iter().zip(&coarse.0) {
            prop_assert!(c.lo <= t.lo && t.hi <= c.hi);
        }
    }
}

proptest! {
    /// `Disjoint` and `Inside` are proofs: no sampled point lies in a polytope
    /// the set was proven disjoint from, and every sampled point lies in one it
    /// was proven to be inside.
    #[test]
    fn polytope_verdicts_are_sound(
        pz in arb_pz(),
        corners in vec((-6.0..6.0f64, 0.0..6.0f64), 3),
        splits in 0usize..12,
        samples in arb_samples(),
    ) {
        let n = pz.dim();
        let b = IntervalBox(corners[..n].iter().map(|&(lo, w)| Interval::new(lo, lo + w)).collect());
        let p = Polytope::from_box(&b);
        let (meets, inside) = (pz.intersection(&p, splits), pz.containment(&p, splits));
        for s in &samples {
            let x = eval(&pz, s);
            let slack = TOL * (1.0 + x.amax());
            if meets == Intersection::Disjoint {
                prop_assert!(!p.contains(&x, -slack));
            }
            if inside == Containment::Inside {
                prop_assert!(p.contains(&x, slack));
            }
        }
    }

    /// A zonotope against a single halfspace needs no splits: both verdicts are
    /// definite and agree with the support function.
    #[test]
    fn zonotope_halfspace_verdicts_are_definite(
        pz in arb_pz(),
        normal in vec(-1.0..1.0f64, 3),
        offset in -10.0..10.0f64,
    ) {
        let z = pz.to_zonotope();
        let a = DVector::from_column_slice(&normal[..z.dim()]);
        let (lowest, highest) = (-support(&z, &(-&a)), support(&z, &a));
        // Skip the measure-zero cases where the halfspace touches the set within rounding.
        prop_assume!((lowest - offset).abs() > 1e-9 && (highest - offset).abs() > 1e-9);
        let h: Polytope = Halfspace { normal: a.clone(), offset }.into();

        let meets = z.intersection(&h, 0);
        prop_assert_eq!(meets, if lowest > offset { Intersection::Disjoint } else { Intersection::Intersects });
        let inside = z.containment(&h, 0);
        prop_assert_eq!(inside, if highest <= offset { Containment::Inside } else { Containment::NotInside });
    }
}

/// The support function `max { dᵀ x | x ∈ Z } = dᵀ c + Σ |dᵀ gⱼ|`.
fn support(z: &Zonotope, d: &DVector<f64>) -> f64 {
    d.dot(z.center())
        + z.generators()
            .column_iter()
            .map(|g| d.dot(&g).abs())
            .sum::<f64>()
}

/// Most generators a reduction to `order` may leave: `order · dim`, plus `dim`
/// rounding-error generators for PCA in sound mode.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "small non-negative counts"
)]
fn max_generators(order: f64, dim: usize, method: Reduction) -> usize {
    let budget = (order * dim as f64) as usize;
    if cfg!(feature = "sound") && method == Reduction::Pca {
        budget + dim
    } else {
        budget
    }
}

/// The coefficients that make `pz.to_zonotope()` hit the point `eval(pz, s)`.
///
/// Mirrors the generator order of [`SparsePolyZonotope::to_zonotope`]: constant
/// monomials fold into the center, even monomials `m ∈ [0, 1]` become `2m - 1`,
/// the rest keep `m`, the independent factors follow, and the coefficients of
/// all-zero generators are left out along with their generators.
fn zonotope_coefficients(pz: &SparsePolyZonotope, s: &Sample) -> Vec<f64> {
    let (e, ids) = (pz.exponents(), pz.ids());
    let a = alpha_map(ids, s);
    let mut coeffs = Vec::new();
    for (col, g) in e.column_iter().zip(pz.dependent_generators().column_iter()) {
        let m = monomial(col.iter().copied(), ids, &a);
        if col.iter().all(|&x| x == 0) || is_zero(g.iter()) {
            continue;
        }

        let even = col.iter().all(|&x| x % 2 == 0);
        coeffs.push(if even { 2.0 * m - 1.0 } else { m });
    }

    let q = pz.independent_generators().ncols();
    assert!(
        s.beta.len() >= q,
        "sample holds {} β values for {q} independent generators",
        s.beta.len()
    );
    let independent = pz.independent_generators().column_iter().zip(&s.beta);
    coeffs.extend(
        independent
            .filter(|(g, _)| !is_zero(g.iter()))
            .map(|(_, &b)| b),
    );
    coeffs
}

/// Whether every entry of a generator column is exactly zero.
fn is_zero<'a>(mut column: impl Iterator<Item = &'a f64>) -> bool {
    column.all(|&x| x == 0.0)
}

/// The monomial `Π_k α_k^exponents[k]` at the factor values `alpha`.
fn monomial(
    exponents: impl Iterator<Item = u32>,
    ids: &[FactorId],
    alpha: &HashMap<FactorId, f64>,
) -> f64 {
    exponents
        .zip(ids)
        .map(|(x, k)| alpha[k].powi(i32::try_from(x).expect("exponents are small")))
        .product()
}

//! Constrained polynomial zonotopes: cuts and intersections keep every feasible
//! point, and sets that cannot meet are proven empty.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{ConstrainedPolyZonotope, Halfspace, IntervalBox, Polytope};
use proptest::collection::vec;
use proptest::prelude::*;

use crate::strategies::{TOL, alpha_map, arb_pz, arb_samples, close, eval, pad};

/// Split budget for the refined hulls in these tests.
const SPLITS: usize = 6;

/// Whether `x` lies in `hull`, with the relative slack [`TOL`].
fn in_hull(hull: &IntervalBox, x: &DVector<f64>) -> bool {
    hull.contains(x, TOL * (1.0 + x.amax()))
}

proptest! {
    /// Cutting a set by a halfspace keeps every sampled point on the inside:
    /// the refined hull contains it, and the cut is never proven empty.
    #[test]
    fn halfspace_cut_keeps_feasible_points(
        pz in arb_pz(),
        normal in vec(-1.0..1.0f64, 3),
        offset in -5.0..5.0f64,
        samples in arb_samples(),
    ) {
        let n = pz.dim();
        let h = Halfspace { normal: DVector::from_column_slice(&normal[..n]), offset };
        let cut = ConstrainedPolyZonotope::from(pz.clone()).intersect_halfspace(&h);
        let hull = cut.interval_hull_split(SPLITS);
        for s in &samples {
            let x = eval(&pz, s);
            if h.normal.dot(&x) <= offset - TOL * (1.0 + x.amax()) {
                let hull = hull.as_ref();
                prop_assert!(hull.is_some_and(|b| in_hull(b, &x)), "feasible point lost: {x}");
            }
        }
    }

    /// The cut is exact and shares the set's factors: for a feasible point,
    /// the slack value that closes `aᵀx + s = d` makes the constraint row
    /// vanish at the same α and β, and the set rows give the point.
    #[test]
    fn halfspace_cut_is_exact(
        pz in arb_pz(),
        normal in vec(-1.0..1.0f64, 3),
        offset in -5.0..5.0f64,
        samples in arb_samples(),
    ) {
        let n = pz.dim();
        let a = DVector::from_column_slice(&normal[..n]);
        let h = Halfspace { normal: a.clone(), offset };
        let cut = ConstrainedPolyZonotope::from(pz.clone()).intersect_halfspace(&h);
        let stacked = cut.stacked();
        let lo = pz.linear_map(&DMatrix::from_row_slice(1, n, a.as_slice())).interval_hull().0[0].lo;
        prop_assume!(cut.num_constraints() == 1 && lo <= offset);
        let half = (offset - lo) / 2.0;
        let slack = stacked.ids().iter().copied().find(|id| !pz.ids().contains(id));
        // The constraint row reuses the set's β; no copies are appended in plain mode.
        if !cfg!(feature = "sound") {
            prop_assert_eq!(stacked.independent_generators().ncols(), pz.independent_generators().ncols());
        }
        for s in &samples {
            let x = eval(&pz, s);
            if a.dot(&x) > offset {
                continue;
            }
            let alpha = alpha_map(pz.ids(), s);
            let closing = (offset - a.dot(&x)) / half - 1.0;
            let value = |id| if Some(id) == slack { closing } else { alpha.get(&id).copied().unwrap_or(0.0) };
            let beta = pad(&s.beta[..pz.independent_generators().ncols()], stacked.independent_generators().ncols());
            let point = stacked.evaluate_with(value, &beta);
            let expected = DVector::from_fn(n + 1, |i, _| if i < n { x[i] } else { 0.0 });

            prop_assert!(close(&point, &expected), "{point} vs {expected}");
        }
    }

    /// A set intersected with a decorrelated copy of itself still contains
    /// every one of its points, and a linear map of the result contains their images.
    #[test]
    fn self_intersection_keeps_every_point(
        pz in arb_pz(),
        m in vec(-2.0..2.0f64, 9),
        samples in arb_samples(),
    ) {
        let n = pz.dim();
        let a = ConstrainedPolyZonotope::from(pz.clone());
        let b = ConstrainedPolyZonotope::from(pz.with_fresh_ids());
        let both = a.intersection(&b);
        let m = DMatrix::from_fn(n, n, |i, j| m[i * 3 + j]);
        let mapped = both.linear_map(&m);
        let (hull, mapped_hull) = (both.interval_hull_split(SPLITS), mapped.interval_hull_split(SPLITS));
        for s in &samples {
            let x = eval(&pz, s);
            prop_assert!(hull.as_ref().is_some_and(|h| in_hull(h, &x)));
            prop_assert!(mapped_hull.as_ref().is_some_and(|h| in_hull(h, &(&m * &x))));
        }
    }

    /// A set and a copy moved beyond its hull cannot meet, and a split-refined
    /// check proves it; the Minkowski sum with a point moves every point.
    #[test]
    fn far_apart_sets_are_proven_disjoint(pz in arb_pz(), samples in arb_samples()) {
        let n = pz.dim();
        let hull = pz.interval_hull();
        let shift = DVector::from_fn(n, |i, _| hull.0[i].width() + 1.0);
        let a = ConstrainedPolyZonotope::from(pz.clone());
        let b = ConstrainedPolyZonotope::from(pz.translate(&shift).with_fresh_ids());

        prop_assert!(a.intersection(&b).proven_empty(SPLITS));

        let summed = a.minkowski_sum(&b).interval_hull_split(SPLITS);
        let moved = a.translate(&shift).interval_hull_split(SPLITS);
        for s in &samples {
            let x = eval(&pz, s);
            prop_assert!(moved.as_ref().is_some_and(|h| in_hull(h, &(&x + &shift))));
            prop_assert!(summed.as_ref().is_some_and(|h| in_hull(h, &(&x * 2.0 + &shift))));
        }
    }
}

/// A box cut by a halfspace beyond it is empty at once; one that keeps a
/// corner shrinks the refined hull towards that corner.
#[test]
fn cutting_a_square() {
    let square = polyzonotope::SparsePolyZonotope::from_zonotope(&polyzonotope::Zonotope::new(
        DVector::zeros(2),
        DMatrix::identity(2, 2),
    ));
    let square = ConstrainedPolyZonotope::from(square);
    let beyond = Halfspace {
        normal: DVector::from_vec(vec![1.0, 0.0]),
        offset: -2.0,
    };
    // x + y >= 1.5 keeps only the corner near (1, 1).
    let corner = Polytope::from(Halfspace {
        normal: DVector::from_vec(vec![-1.0, -1.0]),
        offset: -1.5,
    });

    assert!(square.intersect_halfspace(&beyond).proven_empty(0));
    let cut = square.intersect_polytope(&corner);
    let hull = cut.interval_hull_split(40).expect("not empty");
    // The true hull is [0.5, 1] per axis; the plain hull ignores the cut.
    for (tight, plain) in hull.0.iter().zip(&cut.interval_hull().0) {
        assert!((plain.lo + 1.0).abs() < 1e-12);
        assert!(
            tight.lo >= 0.3 && tight.lo <= 0.5 + 1e-9 && tight.hi >= 1.0 - 1e-9,
            "{tight:?}"
        );
    }
}

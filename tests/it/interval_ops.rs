//! Interval arithmetic and interval-matrix maps: every combination of
//! members lies in the result.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{Interval, IntervalMatrix};
use proptest::collection::vec;
use proptest::prelude::*;

use crate::strategies::{TOL, arb_pz, arb_samples, eval};

/// A random interval within `[-4, 4]`, as `(lo, width)`.
fn arb_interval() -> impl Strategy<Value = Interval> {
    (-4.0..4.0f64, 0.0..2.0f64).prop_map(|(lo, w)| Interval::new(lo, lo + w))
}

/// The point `lo + t (hi - lo)` of `iv`, for `t ∈ [0, 1]`.
fn at(iv: Interval, t: f64) -> f64 {
    iv.lo + t * (iv.hi - iv.lo)
}

/// Whether `x` lies in `iv` up to a relative [`TOL`].
fn holds(iv: Interval, x: f64) -> bool {
    iv.contains(x, TOL * (1.0 + x.abs()))
}

/// A random 3×3 interval matrix and parameters to pick a member of it.
fn arb_matrix() -> impl Strategy<Value = (IntervalMatrix, Vec<f64>)> {
    (vec(arb_interval(), 9), vec(0.0..=1.0f64, 9))
        .prop_map(|(entries, picks)| (IntervalMatrix(DMatrix::from_vec(3, 3, entries)), picks))
}

/// The member of `m` picked by `picks`, one parameter per entry.
fn member(m: &IntervalMatrix, picks: &[f64]) -> DMatrix<f64> {
    DMatrix::from_fn(m.nrows(), m.ncols(), |i, j| {
        at(m.0[(i, j)], picks[i * m.ncols() + j])
    })
}

proptest! {
    /// Sums, differences, products, quotients and negations of members lie in
    /// the interval results.
    #[test]
    fn interval_ops_contain_every_combination(
        a in arb_interval(),
        b in arb_interval(),
        s in 0.0..=1.0f64,
        t in 0.0..=1.0f64,
    ) {
        let (x, y) = (at(a, s), at(b, t));
        prop_assert!(holds(a + b, x + y));
        prop_assert!(holds(a - b, x - y));
        prop_assert!(holds(a * b, x * y));
        prop_assert!(holds(-a, -x));
        prop_assert!(a.mag() >= x.abs());
        let positive = Interval::new(0.5 + b.lo.abs().min(b.hi.abs()), 0.5 + b.mag());
        prop_assert!(holds(a / positive, x / at(positive, t)));
    }

    /// The product of two members lies in the interval product.
    #[test]
    fn interval_matrix_product_contains_member_products(
        (a, pa) in arb_matrix(),
        (b, pb) in arb_matrix(),
    ) {
        let product = a.mul(&b);
        let exact = member(&a, &pa) * member(&b, &pb);
        for i in 0..3 {
            for j in 0..3 {
                prop_assert!(holds(product.0[(i, j)], exact[(i, j)]));
            }
        }
    }

    /// `M x` for a member `M` and a point `x` of the set lies in the hull of the
    /// interval map, for polynomial zonotopes and zonotopes alike.
    #[test]
    fn interval_map_contains_every_image(
        pz in arb_pz(),
        (m, picks) in arb_matrix(),
        samples in arb_samples(),
    ) {
        let n = pz.dim();
        let m = IntervalMatrix(m.0.view((0, 0), (n, n)).into_owned());
        let concrete = member(&m, &picks);
        let (mapped, z_mapped) = (pz.interval_map(&m), pz.to_zonotope().interval_map(&m));
        let (hull, z_hull) = (mapped.interval_hull(), z_mapped.interval_hull());
        for s in &samples {
            let y: DVector<f64> = &concrete * eval(&pz, s);
            prop_assert!(hull.contains(&y, TOL * (1.0 + y.amax())));
            prop_assert!(z_hull.contains(&y, TOL * (1.0 + y.amax())));
        }
    }
}

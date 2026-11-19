//! Zero-tolerance soundness checks for the `sound` feature, on inputs where plain
//! `f64` arithmetic rounds the result to the wrong side.
//!
//! Each check compares the exact real result, an error-free pair from
//! `TwoSum`/`TwoProduct`, against the stored coefficient and the error radius
//! the operation appended, without any further rounding on the way.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, Interval, IntervalBox, Reduction, SparsePolyZonotope, Zonotope};
use proptest::prelude::*;

/// The exact product `a · b` as `p + e`, with `p` rounded and `e` its error.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// The exact sum `a + b` as `s + e`, with `s` rounded and `e` its error (Knuth's `TwoSum`).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Whether the exact number `hi + lo` lies within `r` of `c`, i.e. `|hi + lo - c| <= r`.
///
/// `hi - c` is exact by Sterbenz's lemma, since every caller has `c` within a
/// factor of two of `hi`; the second `two_sum` keeps the rest exact.
#[expect(clippy::float_cmp, reason = "soundness is a zero-tolerance comparison")]
fn within((hi, lo): (f64, f64), c: f64, r: f64) -> bool {
    let (s, t) = two_sum(hi - c, lo);
    s.abs() < r || (s.abs() == r && (t == 0.0 || t.signum() != s.signum()))
}

/// Sum of the absolute values of row 0 of `m`: the error radius of a 1-D result.
fn radius(m: &DMatrix<f64>) -> f64 {
    m.row(0).iter().map(|x| x.abs()).sum()
}

/// The 1-D zonotope `{ c }`.
fn point_zonotope(c: f64) -> Zonotope {
    Zonotope::new(DVector::from_element(1, c), DMatrix::zeros(1, 0))
}

/// The 1-D set `c + Σ g[i] α^e[i]`, all monomials in one fresh factor.
fn one_factor_pz(c: f64, g: &[f64], e: &[u32]) -> SparsePolyZonotope {
    SparsePolyZonotope::new(
        DVector::from_element(1, c),
        DMatrix::from_row_slice(1, g.len(), g),
        DMatrix::zeros(1, 0),
        DMatrix::from_row_slice(1, e.len(), e),
        vec![FactorId::fresh()],
    )
    .expect("consistent shapes")
}

proptest! {
    /// `{ m · c }`: the stored center plus the error generator covers the exact product.
    #[test]
    fn linear_map_encloses_the_exact_product(c in 1.0..1e3f64, m in 1.0..1e3f64) {
        let mapped = point_zonotope(c).linear_map(&DMatrix::from_element(1, 1, m));

        prop_assert!(within(two_product(m, c), mapped.center()[0], radius(mapped.generators())));
    }

    /// `{ c + v }`: the stored center plus the error generator covers the exact sum.
    #[test]
    fn translate_encloses_the_exact_sum(c in 1.0..1e3f64, v in 1e-6..1e-3f64) {
        let moved = point_zonotope(c).translate(&DVector::from_element(1, v));

        prop_assert!(within(two_sum(c, v), moved.center()[0], radius(moved.generators())));
    }

    /// Both bounds of an interval sum enclose the exact sums of the bounds.
    #[test]
    fn interval_sum_is_rounded_outward(a in -1e3..1e3f64, b in -1e-3..1e-3f64) {
        let sum = Interval::point(a) + Interval::point(b);
        let (s, e) = two_sum(a, b);

        prop_assert!(sum.lo < s || (sum.lo <= s && e >= 0.0));
        prop_assert!(sum.hi > s || (sum.hi >= s && e <= 0.0));
    }

    /// The ball built by `from_box` covers both bounds of the box.
    #[test]
    fn from_box_covers_the_box(lo in 1.0..1e3f64, w in 0.0..1.0f64) {
        let b = Interval::new(lo, lo + w);
        let z = Zonotope::from_box(&IntervalBox(vec![b]));
        let (c, r) = (z.center()[0], z.generators()[(0, 0)]);

        prop_assert!(within((b.lo, 0.0), c, r) && within((b.hi, 0.0), c, r));
    }

    /// `exact_plus` of two points: the center plus the error generator covers the exact sum.
    #[test]
    fn exact_plus_encloses_the_exact_sum(a in 1.0..1e3f64, b in 1e-6..1e-3f64) {
        let sum = one_factor_pz(a, &[], &[]).exact_plus(&one_factor_pz(b, &[], &[]));

        prop_assert!(within(two_sum(a, b), sum.center()[0], radius(sum.independent_generators())));
    }

    /// `compact` merging two generators with the same exponent covers their exact sum.
    #[test]
    fn compact_encloses_the_merged_generator(a in 1.0..1e3f64, b in 1e-6..1e-3f64) {
        let merged = one_factor_pz(0.0, &[a, b], &[1, 1]).compact();

        prop_assert_eq!(merged.dependent_generators().ncols(), 1);
        let g = merged.dependent_generators()[(0, 0)];
        prop_assert!(within(two_sum(a, b), g, radius(merged.independent_generators())));
    }

    /// Splitting `g α³` gives the coefficient `3g/8` for `α'`; the stored value
    /// plus the error generators covers it.
    #[test]
    fn split_encloses_the_rescaled_coefficient(g in 1.0..1e3f64) {
        let pz = one_factor_pz(0.0, &[g], &[3]);
        let (lower, _) = pz.split(pz.ids()[0]);
        let linear = lower
            .exponents()
            .column_iter()
            .position(|e| e[0] == 1)
            .expect("the expansion has a linear term");
        let stored = lower.dependent_generators()[(0, linear)];

        prop_assert!(within(two_product(g, 0.375), stored, radius(lower.independent_generators())));
    }
}

proptest! {
    /// `{ c² }` as the quadratic map of the point `c` with `Q = [1]`: the
    /// stored center plus the error generator covers the exact square.
    #[test]
    fn quadratic_map_encloses_the_exact_square(c in 1.0..1e3f64) {
        let mapped = one_factor_pz(c, &[], &[]).quadratic_map(&[DMatrix::from_element(1, 1, 1.0)]);

        prop_assert!(within(two_product(c, c), mapped.center()[0], radius(mapped.independent_generators())));
    }
}

proptest! {
    /// Girard's box of two parallel 1-D generators `a` and `b` has radius at
    /// least `|a| + |b|`, exactly, after reducing to order 1.
    #[test]
    fn girard_box_covers_the_exact_radius(a in 1.0..1e3f64, b in 1e-6..1e-3f64) {
        let z = Zonotope::new(DVector::zeros(1), DMatrix::from_row_slice(1, 2, &[a, b]));
        let reduced = z.reduce(1.0, Reduction::Girard);

        prop_assert_eq!(reduced.num_generators(), 1);
        prop_assert!(within(two_sum(a, b), 0.0, radius(reduced.generators())));
    }
}

/// `3 · 0.1` rounds up in `f64`, so the plain center misses the exact product;
/// the sound map adds an error generator that covers it.
#[test]
fn inexact_map_gets_an_error_generator() {
    let mapped = point_zonotope(0.1).linear_map(&DMatrix::from_element(1, 1, 3.0));
    let exact = two_product(3.0, 0.1);

    assert!(exact.1 != 0.0, "the test needs an inexact product");
    assert_eq!(mapped.num_generators(), 1);
    assert!(within(
        exact,
        mapped.center()[0],
        radius(mapped.generators())
    ));
}

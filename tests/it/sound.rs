//! Zero-tolerance soundness checks for the `sound` feature, on inputs where plain
//! `f64` arithmetic rounds the result to the wrong side.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{Interval, IntervalBox, Zonotope};
use proptest::prelude::*;

/// Whether the exact real number `hi + lo` (an error-free pair) lies in `iv`.
#[expect(clippy::float_cmp, reason = "soundness is a zero-tolerance comparison")]
fn contains_exact(iv: Interval, hi: f64, lo: f64) -> bool {
    let above_lower = iv.lo < hi || (iv.lo == hi && lo >= 0.0);
    let below_upper = iv.hi > hi || (iv.hi == hi && lo <= 0.0);

    above_lower && below_upper
}

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

/// The single interval of a one-dimensional set's hull.
fn hull_1d(z: &Zonotope) -> Interval {
    z.interval_hull().0[0]
}

proptest! {
    /// `{ m · c }` maps to an interval that contains the exact product.
    #[test]
    fn linear_map_encloses_the_exact_product(c in -1e3..1e3f64, m in -1e3..1e3f64) {
        let z = Zonotope::new(DVector::from_element(1, c), DMatrix::zeros(1, 0));
        let mapped = z.linear_map(&DMatrix::from_element(1, 1, m));
        let (p, e) = two_product(m, c);

        prop_assert!(contains_exact(hull_1d(&mapped), p, e));
    }

    /// `{ c + v }` translates to an interval that contains the exact sum.
    #[test]
    fn translate_encloses_the_exact_sum(c in -1e3..1e3f64, v in -1e-3..1e-3f64) {
        let z = Zonotope::new(DVector::from_element(1, c), DMatrix::zeros(1, 0));
        let moved = z.translate(&DVector::from_element(1, v));
        let (s, e) = two_sum(c, v);

        prop_assert!(contains_exact(hull_1d(&moved), s, e));
    }

    /// Both bounds of an interval sum enclose the exact sums of the bounds.
    #[test]
    fn interval_sum_is_rounded_outward(a in -1e3..1e3f64, b in -1e-3..1e-3f64) {
        let sum = Interval::point(a) + Interval::point(b);
        let (s, e) = two_sum(a, b);

        prop_assert!(contains_exact(sum, s, e));
    }

    /// The box built by `from_box` covers both bounds of the original box.
    #[test]
    fn from_box_covers_the_box(lo in -1e3..1e3f64, w in 0.0..1e3f64) {
        let b = Interval::new(lo, lo + w);
        let z = Zonotope::from_box(&IntervalBox(vec![b]));
        let (c, r) = (z.center()[0], z.generators()[(0, 0)]);

        prop_assert!(contains_exact(Interval::point(c) + Interval::new(-r, r), b.lo, 0.0));
        prop_assert!(contains_exact(Interval::point(c) + Interval::new(-r, r), b.hi, 0.0));
    }
}

/// `3 · 0.1` rounds up in `f64`, so the plain center misses the exact product;
/// the sound map adds an error generator that covers it.
#[test]
fn inexact_map_gets_an_error_generator() {
    let z = Zonotope::new(DVector::from_element(1, 0.1), DMatrix::zeros(1, 0));
    let mapped = z.linear_map(&DMatrix::from_element(1, 1, 3.0));
    let (p, e) = two_product(3.0, 0.1);

    assert!(e != 0.0, "the test needs an inexact product");
    assert_eq!(mapped.num_generators(), 1);
    assert!(contains_exact(hull_1d(&mapped), p, e));
}

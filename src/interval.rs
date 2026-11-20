//! Closed intervals and axis-aligned interval boxes, the coarsest set representation.

use alloc::vec::Vec;

use nalgebra::DVector;

use crate::round;

/// Closed interval `[lo, hi]` of reals.
///
/// Invariant: `lo <= hi` when built through [`Interval::new`] or [`Interval::point`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Lower bound, inclusive.
    pub lo: f64,
    /// Upper bound, inclusive; never below [`lo`](Self::lo).
    pub hi: f64,
}

impl Interval {
    /// The interval `[lo, hi]`.
    ///
    /// # Panics
    ///
    /// Panics if `lo > hi` (the interval would be empty), or if either bound is NaN.
    #[must_use]
    pub fn new(lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "empty interval [{lo}, {hi}]");
        Self { lo, hi }
    }

    /// The degenerate interval `[x, x]`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is NaN, like [`new`](Self::new) does.
    #[must_use]
    pub fn point(x: f64) -> Self {
        assert!(!x.is_nan(), "interval bound is NaN");
        Self { lo: x, hi: x }
    }

    /// Length `hi - lo`; zero for a point.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Largest absolute value in the interval, `max(|lo|, |hi|)`; exact.
    #[must_use]
    pub fn mag(&self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    /// Whether `x` lies in the interval widened by `tol` on both sides.
    #[must_use]
    pub fn contains(&self, x: f64, tol: f64) -> bool {
        self.lo - tol <= x && x <= self.hi + tol
    }
}

impl core::ops::Add for Interval {
    type Output = Interval;

    /// Minkowski sum `{ a + b | a ∈ self, b ∈ o }`, which is `[lo + o.lo, hi + o.hi]`.
    ///
    /// With the `sound` feature the bounds are rounded outward.
    fn add(self, o: Interval) -> Interval {
        Interval {
            lo: round::add_down(self.lo, o.lo),
            hi: round::add_up(self.hi, o.hi),
        }
    }
}

/// Axis-aligned box: one interval per dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalBox(pub Vec<Interval>);

impl IntervalBox {
    /// Number of dimensions, one per interval.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Whether `p` has the box's dimension and lies in it, widened by `tol` per axis.
    #[must_use]
    pub fn contains(&self, p: &DVector<f64>, tol: f64) -> bool {
        p.len() == self.dim()
            && self
                .0
                .iter()
                .zip(p.iter())
                .all(|(iv, &x)| iv.contains(x, tol))
    }
}

impl core::ops::Neg for Interval {
    type Output = Interval;

    /// `{ -a | a ∈ self }`, which is `[-hi, -lo]`; exact.
    fn neg(self) -> Interval {
        Interval {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl core::ops::Sub for Interval {
    type Output = Interval;

    /// `{ a - b | a ∈ self, b ∈ o }`, which is `[lo - o.hi, hi - o.lo]`.
    ///
    /// With the `sound` feature the bounds are rounded outward.
    fn sub(self, o: Interval) -> Interval {
        self + (-o)
    }
}

impl core::ops::Mul for Interval {
    type Output = Interval;

    /// `{ a · b | a ∈ self, b ∈ o }`, from the extreme corner products.
    ///
    /// With the `sound` feature the bounds are rounded outward.
    fn mul(self, o: Interval) -> Interval {
        let (lo, hi) = round::mul_bounds((self.lo, self.hi), (o.lo, o.hi));
        Interval { lo, hi }
    }
}

impl core::ops::Div for Interval {
    type Output = Interval;

    /// `{ a / b | a ∈ self, b ∈ o }`.
    ///
    /// With the `sound` feature the bounds are rounded outward.
    ///
    /// # Panics
    ///
    /// Panics if the divisor contains zero.
    fn div(self, o: Interval) -> Interval {
        assert!(
            o.lo > 0.0 || o.hi < 0.0,
            "interval division by an interval containing zero"
        );
        let (lo, hi) = round::div_bounds((self.lo, self.hi), (o.lo, o.hi));
        Interval { lo, hi }
    }
}

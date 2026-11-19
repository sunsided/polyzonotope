//! Rounding-error bookkeeping for the set operations.
//!
//! Every operation that computes new coefficients goes through these helpers.
//! Each one returns the rounded `f64` result together with a bound on how far
//! the real result can lie from it. With the `sound` feature the bounds come
//! from [`inari`]'s outward-rounded interval arithmetic. Without it they are
//! zero, so plain mode adds no generators and multiplies matrices with nalgebra.
//!
//! Operations collect the bounds per dimension in [`RowErrors`] and append
//! them to the set as axis-aligned generators, so the stored set encloses the
//! set the real arithmetic would have produced.

use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};

#[cfg(feature = "sound")]
use sound as backend;

#[cfg(not(feature = "sound"))]
use plain as backend;

pub(crate) use backend::{add_down, add_up, mid_rad};

/// Outward-rounded arithmetic on top of [`inari`].
#[cfg(feature = "sound")]
mod sound {
    use inari::Interval;
    use nalgebra::{DMatrix, DVector};

    use super::RowErrors;

    /// The degenerate interval `[x, x]`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is NaN or infinite, which inari does not accept as a bound
    /// of a point interval.
    fn point(x: f64) -> Interval {
        Interval::try_from((x, x)).expect("set coefficients must be finite in sound mode")
    }

    /// Midpoint of `x` and the smallest radius around it that still covers `x`.
    fn split(x: Interval) -> (f64, f64) {
        (x.mid(), x.rad())
    }

    /// `a + b` rounded to nearest, and a bound on its error.
    pub(crate) fn add(a: f64, b: f64) -> (f64, f64) {
        split(point(a) + point(b))
    }

    /// `a · b` rounded to nearest, and a bound on its error.
    pub(crate) fn mul(a: f64, b: f64) -> (f64, f64) {
        split(point(a) * point(b))
    }

    /// `Σ aᵢ bᵢ`, and a bound on its error.
    fn dot(pairs: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
        split(pairs.fold(point(0.0), |acc, (a, b)| point(a).mul_add(point(b), acc)))
    }

    /// An upper bound of `a + b`.
    pub(crate) fn add_up(a: f64, b: f64) -> f64 {
        (point(a) + point(b)).sup()
    }

    /// A lower bound of `a + b`.
    pub(crate) fn add_down(a: f64, b: f64) -> f64 {
        (point(a) + point(b)).inf()
    }

    /// Center and radius of a ball that covers `[lo, hi]`.
    pub(crate) fn mid_rad(lo: f64, hi: f64) -> (f64, f64) {
        split(Interval::try_from((lo, hi)).expect("interval bounds must be finite in sound mode"))
    }

    /// `M X`, with the rounding error of each entry recorded on its row.
    pub(crate) fn mat_mul(
        m: &DMatrix<f64>,
        x: &DMatrix<f64>,
        errs: &mut RowErrors,
    ) -> DMatrix<f64> {
        DMatrix::from_fn(m.nrows(), x.ncols(), |i, j| {
            let (v, e) = dot(m.row(i).iter().copied().zip(x.column(j).iter().copied()));
            errs.record(i, e);
            v
        })
    }

    /// `M v`, with the rounding error of each entry recorded on its row.
    pub(crate) fn mat_vec(
        m: &DMatrix<f64>,
        v: &DVector<f64>,
        errs: &mut RowErrors,
    ) -> DVector<f64> {
        DVector::from_fn(m.nrows(), |i, _| {
            let (r, e) = dot(m.row(i).iter().copied().zip(v.iter().copied()));
            errs.record(i, e);
            r
        })
    }
}

/// Plain `f64` arithmetic: every error bound is zero.
#[cfg(not(feature = "sound"))]
mod plain {
    use nalgebra::{DMatrix, DVector};

    use super::RowErrors;

    /// `a + b`, with no error bound.
    pub(crate) fn add(a: f64, b: f64) -> (f64, f64) {
        (a + b, 0.0)
    }

    /// `a · b`, with no error bound.
    pub(crate) fn mul(a: f64, b: f64) -> (f64, f64) {
        (a * b, 0.0)
    }

    /// `a + b`, taken as its own upper bound.
    pub(crate) fn add_up(a: f64, b: f64) -> f64 {
        a + b
    }

    /// `a + b`, taken as its own lower bound.
    pub(crate) fn add_down(a: f64, b: f64) -> f64 {
        a + b
    }

    /// Midpoint and half width of `[lo, hi]`.
    pub(crate) fn mid_rad(lo: f64, hi: f64) -> (f64, f64) {
        (f64::midpoint(lo, hi), (hi - lo) / 2.0)
    }

    /// `M X` by nalgebra.
    pub(crate) fn mat_mul(m: &DMatrix<f64>, x: &DMatrix<f64>, _: &mut RowErrors) -> DMatrix<f64> {
        m * x
    }

    /// `M v` by nalgebra.
    pub(crate) fn mat_vec(m: &DMatrix<f64>, v: &DVector<f64>, _: &mut RowErrors) -> DVector<f64> {
        m * v
    }
}

/// Accumulated rounding-error bound per dimension of a set under construction.
pub(crate) struct RowErrors(Vec<f64>);

impl RowErrors {
    /// No error yet, for a set with `n` dimensions.
    pub(crate) fn new(n: usize) -> Self {
        Self(alloc::vec![0.0; n])
    }

    /// Adds `err` to the bound of dimension `row`, rounding the sum up.
    pub(crate) fn record(&mut self, row: usize, err: f64) {
        if err != 0.0 {
            self.0[row] = add_up(self.0[row], err);
        }
    }

    /// The bounds as generators: one axis-aligned column per dimension with a
    /// nonzero bound, none at all in plain mode.
    pub(crate) fn generators(&self) -> DMatrix<f64> {
        let rows: Vec<usize> = (0..self.0.len()).filter(|&i| self.0[i] != 0.0).collect();
        DMatrix::from_fn(self.0.len(), rows.len(), |i, j| {
            if i == rows[j] { self.0[i] } else { 0.0 }
        })
    }
}

/// `M X`, with the rounding error of each entry recorded on its row of `errs`.
///
/// # Panics
///
/// Panics if `m` does not have one column per row of `x`.
pub(crate) fn mat_mul(m: &DMatrix<f64>, x: &DMatrix<f64>, errs: &mut RowErrors) -> DMatrix<f64> {
    assert_eq!(
        m.ncols(),
        x.nrows(),
        "matrix columns must match the dimension"
    );
    backend::mat_mul(m, x, errs)
}

/// `M v`, with the rounding error of each entry recorded on its row of `errs`.
///
/// # Panics
///
/// Panics if `m` does not have one column per entry of `v`.
pub(crate) fn mat_vec(m: &DMatrix<f64>, v: &DVector<f64>, errs: &mut RowErrors) -> DVector<f64> {
    assert_eq!(
        m.ncols(),
        v.len(),
        "matrix columns must match the dimension"
    );
    backend::mat_vec(m, v, errs)
}

/// `a + b`, with the rounding error of each entry recorded in `errs`.
///
/// # Panics
///
/// Panics if `a` and `b` differ in length.
pub(crate) fn vec_add(a: &DVector<f64>, b: &DVector<f64>, errs: &mut RowErrors) -> DVector<f64> {
    let mut sum = a.clone();
    add_into(&mut sum, b, errs);
    sum
}

/// `target += v`, with the rounding error of each entry recorded in `errs`.
///
/// # Panics
///
/// Panics if `target` and `v` differ in length.
pub(crate) fn add_into(target: &mut DVector<f64>, v: &DVector<f64>, errs: &mut RowErrors) {
    assert_eq!(
        target.len(),
        v.len(),
        "vectors must have the same dimension"
    );
    for (i, (t, &x)) in target.iter_mut().zip(v.iter()).enumerate() {
        let (sum, e) = backend::add(*t, x);
        errs.record(i, e);
        *t = sum;
    }
}

/// `s · v`, with the rounding error of each entry recorded in `errs`.
pub(crate) fn scaled(v: &DVector<f64>, s: f64, errs: &mut RowErrors) -> DVector<f64> {
    DVector::from_fn(v.len(), |i, _| {
        let (p, e) = backend::mul(v[i], s);
        errs.record(i, e);
        p
    })
}

/// An upper bound of `Σ |xᵢ|`.
pub(crate) fn abs_sum_up<'a>(xs: impl IntoIterator<Item = &'a f64>) -> f64 {
    xs.into_iter().fold(0.0, |acc, x| add_up(acc, x.abs()))
}

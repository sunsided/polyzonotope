//! Matrices with interval entries, for linear maps with uncertain or
//! rounding-afflicted coefficients such as a truncated matrix exponential.

use alloc::vec::Vec;

use nalgebra::DMatrix;

use crate::{Interval, IntervalBox, round};

/// A matrix whose entries are intervals: the set of all real matrices with
/// each entry in its interval.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalMatrix(pub DMatrix<Interval>);

impl IntervalMatrix {
    /// The degenerate interval matrix holding only `m`.
    ///
    /// # Panics
    ///
    /// Panics if an entry of `m` is NaN.
    #[must_use]
    pub fn point(m: &DMatrix<f64>) -> Self {
        Self(m.map(Interval::point))
    }

    /// The zero matrix of the given shape.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self(DMatrix::from_element(rows, cols, Interval::point(0.0)))
    }

    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.0.nrows()
    }

    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.0.ncols()
    }

    /// `{ A + B | A ∈ self, B ∈ o }`, entrywise.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[must_use]
    pub fn add(&self, o: &Self) -> Self {
        assert_eq!(
            self.0.shape(),
            o.0.shape(),
            "interval matrices must have the same shape"
        );
        Self(self.0.zip_map(&o.0, |a, b| a + b))
    }

    /// An enclosure of `{ A B | A ∈ self, B ∈ o }`: each entry is the
    /// interval sum of interval products.
    ///
    /// # Panics
    ///
    /// Panics if `self` does not have one column per row of `o`.
    #[must_use]
    pub fn mul(&self, o: &Self) -> Self {
        assert_eq!(
            self.ncols(),
            o.nrows(),
            "matrix columns must match the rows of the other"
        );
        Self(DMatrix::from_fn(self.nrows(), o.ncols(), |i, j| {
            (0..self.ncols()).fold(Interval::point(0.0), |acc, k| {
                acc + self.0[(i, k)] * o.0[(k, j)]
            })
        }))
    }

    /// `{ s A | s ∈ factor, A ∈ self }`, entrywise.
    #[must_use]
    pub fn scale(&self, factor: Interval) -> Self {
        Self(self.0.map(|a| a * factor))
    }

    /// Center `C` and radius `R >= 0` with every matrix of the set inside
    /// `C ± R` entrywise; `R` is rounded up with the `sound` feature.
    #[must_use]
    pub fn center_radius(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let parts = self.0.map(|a| round::mid_rad(a.lo, a.hi));
        (parts.map(|(c, _)| c), parts.map(|(_, r)| r))
    }
}

/// Axis-aligned generators that cover `{ (A - C) x | A ∈ C ± R, x ∈ hull }`:
/// one column per output row `i` with a nonzero radius `Σ_j R_ij max|x_j|`,
/// rounded up with the `sound` feature.
pub(crate) fn uncertainty_generators(radius: &DMatrix<f64>, hull: &IntervalBox) -> DMatrix<f64> {
    let reach: Vec<f64> = (0..radius.nrows())
        .map(|i| {
            (0..radius.ncols())
                .fold(Interval::point(0.0), |acc, j| {
                    acc + Interval::point(radius[(i, j)]) * Interval::point(hull.0[j].mag())
                })
                .hi
        })
        .collect();
    let rows: Vec<usize> = (0..reach.len()).filter(|&i| reach[i] != 0.0).collect();

    DMatrix::from_fn(reach.len(), rows.len(), |i, j| {
        if i == rows[j] { reach[i] } else { 0.0 }
    })
}

//! Zonotopes: affine images of a hypercube, closed under linear maps and Minkowski sums.

use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};

use crate::round::{self, RowErrors};
use crate::{Interval, IntervalBox, hcat};

/// `Z = { c + G β | β ∈ [-1, 1]^m }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonotope {
    /// Center `c`; its length is the dimension of the set.
    center: DVector<f64>,
    /// `dim × m` generator matrix; one row per entry of [`center`](Self::center).
    generators: DMatrix<f64>,
}

impl Zonotope {
    /// The zonotope with the given center and generators (one per column).
    ///
    /// # Panics
    ///
    /// Panics if `generators` does not have one row per entry of `center`.
    #[must_use]
    pub fn new(center: DVector<f64>, generators: DMatrix<f64>) -> Self {
        assert_eq!(
            center.len(),
            generators.nrows(),
            "generator rows must match dimension"
        );
        Self { center, generators }
    }

    /// Exact conversion of a box: its midpoint as center, half widths on the diagonal.
    ///
    /// With the `sound` feature the half widths are rounded up, so the result
    /// covers the box even when the midpoint is not representable.
    #[must_use]
    pub fn from_box(b: &IntervalBox) -> Self {
        let (c, r): (Vec<f64>, Vec<f64>) = b.0.iter().map(|i| round::mid_rad(i.lo, i.hi)).unzip();

        Self::new(
            DVector::from_vec(c),
            DMatrix::from_diagonal(&DVector::from_vec(r)),
        )
    }

    /// The zonotope with center `center` and generators `generators`, plus one
    /// axis-aligned generator per nonzero rounding-error bound in `errs`.
    fn with_errors(center: DVector<f64>, generators: &DMatrix<f64>, errs: &RowErrors) -> Self {
        Self::new(center, hcat(generators, &errs.generators()))
    }

    /// Center `c`.
    #[must_use]
    pub fn center(&self) -> &DVector<f64> {
        &self.center
    }

    /// Generator matrix `G`, one generator per column and one row per dimension.
    #[must_use]
    pub fn generators(&self) -> &DMatrix<f64> {
        &self.generators
    }

    /// Dimension of the space the set lives in.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.center.len()
    }

    /// Number of generators `m`, the columns of [`generators`](Self::generators).
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.generators.ncols()
    }

    /// `m / n`, the usual measure of representation size.
    ///
    /// # Panics
    ///
    /// Panics if the set is zero-dimensional, where the order is undefined.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "generator counts and dimensions stay far below 2^52"
    )]
    pub fn order(&self) -> f64 {
        assert!(self.dim() > 0, "order of a zero-dimensional zonotope");
        self.num_generators() as f64 / self.dim() as f64
    }

    /// The point `c + G β` for the factor values `beta`, one per generator.
    ///
    /// # Panics
    ///
    /// Panics if `beta` does not have one entry per generator.
    #[must_use]
    pub fn evaluate(&self, beta: &[f64]) -> DVector<f64> {
        &self.center + &self.generators * DVector::from_column_slice(beta)
    }

    /// Exact: `{ M x | x ∈ Z }`.
    ///
    /// # Panics
    ///
    /// Panics if `m` does not have one column per dimension.
    #[must_use]
    pub fn linear_map(&self, m: &DMatrix<f64>) -> Self {
        let mut errs = RowErrors::new(m.nrows());
        let c = round::mat_vec(m, &self.center, &mut errs);
        let g = round::mat_mul(m, &self.generators, &mut errs);

        Self::with_errors(c, &g, &errs)
    }

    /// Exact: `{ x + v | x ∈ Z }`.
    ///
    /// # Panics
    ///
    /// Panics if `v` does not have the set's dimension.
    #[must_use]
    pub fn translate(&self, v: &DVector<f64>) -> Self {
        let mut errs = RowErrors::new(self.dim());
        let c = round::vec_add(&self.center, v, &mut errs);

        Self::with_errors(c, &self.generators, &errs)
    }

    /// Exact Minkowski sum `{ x + y | x ∈ self, y ∈ o }`: centers add, generators concatenate.
    ///
    /// # Panics
    ///
    /// Panics if the two zonotopes have different dimensions.
    #[must_use]
    pub fn minkowski_sum(&self, o: &Self) -> Self {
        let mut errs = RowErrors::new(self.dim());
        let c = round::vec_add(&self.center, &o.center, &mut errs);

        Self::with_errors(c, &hcat(&self.generators, &o.generators), &errs)
    }

    /// Tight interval enclosure (rounded outward with the `sound` feature).
    #[must_use]
    pub fn interval_hull(&self) -> IntervalBox {
        IntervalBox(
            (0..self.dim())
                .map(|i| {
                    let r = round::abs_sum_up(self.generators.row(i).iter());
                    Interval::point(self.center[i]) + Interval::new(-r, r)
                })
                .collect(),
        )
    }
}

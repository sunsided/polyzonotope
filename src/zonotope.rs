//! Zonotopes: affine images of a hypercube, closed under linear maps and Minkowski sums.

use nalgebra::{DMatrix, DVector};

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
    #[must_use]
    pub fn from_box(b: &IntervalBox) -> Self {
        let c = DVector::from_iterator(b.dim(), b.0.iter().map(|i| f64::midpoint(i.lo, i.hi)));
        let g = DMatrix::from_diagonal(&DVector::from_iterator(
            b.dim(),
            b.0.iter().map(|i| i.width() / 2.0),
        ));

        Self::new(c, g)
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
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "generator counts and dimensions stay far below 2^52"
    )]
    pub fn order(&self) -> f64 {
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
        Self::new(m * &self.center, m * &self.generators)
    }

    /// Exact: `{ x + v | x ∈ Z }`.
    ///
    /// # Panics
    ///
    /// Panics if `v` does not have the set's dimension.
    #[must_use]
    pub fn translate(&self, v: &DVector<f64>) -> Self {
        Self::new(&self.center + v, self.generators.clone())
    }

    /// Exact Minkowski sum `{ x + y | x ∈ self, y ∈ o }`: centers add, generators concatenate.
    ///
    /// # Panics
    ///
    /// Panics if the two zonotopes have different dimensions.
    #[must_use]
    pub fn minkowski_sum(&self, o: &Self) -> Self {
        Self::new(
            &self.center + &o.center,
            hcat(&self.generators, &o.generators),
        )
    }

    /// Tight interval enclosure.
    #[must_use]
    pub fn interval_hull(&self) -> IntervalBox {
        IntervalBox(
            (0..self.dim())
                .map(|i| {
                    let r: f64 = self.generators.row(i).iter().map(|g| g.abs()).sum();
                    Interval::new(self.center[i] - r, self.center[i] + r)
                })
                .collect(),
        )
    }
}

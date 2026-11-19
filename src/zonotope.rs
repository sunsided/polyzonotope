use nalgebra::{DMatrix, DVector};

use crate::{Interval, IntervalBox, hcat};

/// `Z = { c + G β | β ∈ [-1, 1]^m }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonotope {
    pub center: DVector<f64>,
    /// `dim × m` generator matrix.
    pub generators: DMatrix<f64>,
}

impl Zonotope {
    pub fn new(center: DVector<f64>, generators: DMatrix<f64>) -> Self {
        assert_eq!(
            center.len(),
            generators.nrows(),
            "generator rows must match dimension"
        );
        Self { center, generators }
    }

    pub fn from_box(b: &IntervalBox) -> Self {
        let c = DVector::from_iterator(b.dim(), b.0.iter().map(|i| (i.lo + i.hi) / 2.0));
        let g = DMatrix::from_diagonal(&DVector::from_iterator(
            b.dim(),
            b.0.iter().map(|i| i.width() / 2.0),
        ));
        Self::new(c, g)
    }

    pub fn dim(&self) -> usize {
        self.center.len()
    }

    pub fn num_generators(&self) -> usize {
        self.generators.ncols()
    }

    /// `m / n`, the usual measure of representation size.
    pub fn order(&self) -> f64 {
        self.num_generators() as f64 / self.dim() as f64
    }

    pub fn evaluate(&self, beta: &[f64]) -> DVector<f64> {
        &self.center + &self.generators * DVector::from_column_slice(beta)
    }

    pub fn linear_map(&self, m: &DMatrix<f64>) -> Self {
        Self::new(m * &self.center, m * &self.generators)
    }

    pub fn translate(&self, v: &DVector<f64>) -> Self {
        Self::new(&self.center + v, self.generators.clone())
    }

    pub fn minkowski_sum(&self, o: &Self) -> Self {
        Self::new(
            &self.center + &o.center,
            hcat(&self.generators, &o.generators),
        )
    }

    /// Tight interval enclosure.
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

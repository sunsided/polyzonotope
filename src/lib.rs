#![doc = include_str!("../README.md")]

mod interval;
mod sparse;
mod zonotope;

pub use interval::{Interval, IntervalBox};
pub use sparse::{FactorId, ShapeError, SparsePolyZonotope};
pub use zonotope::Zonotope;

use nalgebra::DMatrix;

/// Horizontal concatenation that tolerates zero-column inputs.
pub(crate) fn hcat<T: nalgebra::Scalar>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T> {
    assert_eq!(a.nrows(), b.nrows());
    DMatrix::from_fn(a.nrows(), a.ncols() + b.ncols(), |i, j| {
        if j < a.ncols() {
            a[(i, j)].clone()
        } else {
            b[(i, j - a.ncols())].clone()
        }
    })
}

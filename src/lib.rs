#![doc = include_str!("../README.md")]
#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod constrained;
mod interval;
mod interval_matrix;
mod polytope;
mod reduce;
mod round;
mod sparse;
mod zonotope;

pub use constrained::ConstrainedPolyZonotope;
pub use interval::{Interval, IntervalBox};
pub use interval_matrix::IntervalMatrix;
pub use polytope::{Containment, Halfspace, Intersection, Polytope};
pub use reduce::Reduction;
pub use sparse::{FactorId, ShapeError, SparsePolyZonotope};
pub use zonotope::Zonotope;

use core::ops::Index;

use nalgebra::DMatrix;

/// The `nrows × cols.len()` matrix whose column `j` is `cols[j]`.
///
/// Unlike [`DMatrix::from_columns`], this accepts an empty list, which gives
/// a matrix with zero columns.
pub(crate) fn from_column_list<T, C>(nrows: usize, cols: &[C]) -> DMatrix<T>
where
    T: nalgebra::Scalar,
    C: Index<usize, Output = T>,
{
    DMatrix::from_fn(nrows, cols.len(), |i, j| cols[j][i].clone())
}

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

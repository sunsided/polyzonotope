//! Set representations for reachability analysis: [`IntervalBox`],
//! [`Zonotope`], and [`SparsePolyZonotope`] (Kochdumper & Althoff, 2021).
//!
//! A sparse polynomial zonotope is the set
//!
//! ```text
//! PZ = { c + Σ_i (Π_k α_k^E[k,i]) G[:,i] + Σ_j β_j GI[:,j]  |  α, β ∈ [-1, 1] }
//! ```
//!
//! Dependent factors `α_k` carry global [`FactorId`]s, so two sets built from
//! the same uncertainty stay correlated through later operations.
//!
//! **Soundness caveat:** this version uses plain `f64` arithmetic. Enclosures
//! are mathematically sound, but rounding is not yet outward-directed; a
//! sound mode (interval arithmetic via `inari`) is on the roadmap.

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

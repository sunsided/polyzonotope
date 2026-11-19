//! Proptest strategies for random sets and factor samples, plus evaluation helpers.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, SparsePolyZonotope};
use proptest::collection::vec;
use proptest::prelude::*;

/// Relative tolerance for comparing points computed along different paths.
pub const TOL: f64 = 1e-9;

/// Number of α and β values in every [`Sample`]; at least the largest factor
/// and independent-generator count [`arb_pz`] produces, plus one spare α for `split`.
pub const SAMPLE_LEN: usize = 8;

/// Random sparse polynomial zonotope with 1–3 dimensions, up to 5 dependent
/// and 2 independent generators (some of them zero), 1–3 fresh factors and
/// exponents in `0..=3`.
pub fn arb_pz() -> impl Strategy<Value = SparsePolyZonotope> {
    (1usize..=3, 0usize..=5, 0usize..=2, 1usize..=3).prop_flat_map(|(n, h, q, p)| {
        (
            vec(-5.0..5.0f64, n),
            vec(-2.0..2.0f64, n * h),
            // Exact zeros make whole zero columns likely: compact() must keep them so β positions stay stable.
            vec(prop_oneof![1 => Just(0.0), 3 => -2.0..2.0f64], n * q),
            vec(0u32..=3, p * h),
        )
            .prop_map(move |(c, g, gi, e)| {
                SparsePolyZonotope::new(
                    DVector::from_vec(c),
                    DMatrix::from_vec(n, h, g),
                    DMatrix::from_vec(n, q, gi),
                    DMatrix::from_vec(p, h, e),
                    FactorId::fresh_n(p),
                )
                .expect("the strategy builds consistent shapes")
            })
    })
}

/// Random factor values: `alpha[k]` belongs to the `k`-th id of a set, and
/// `beta` holds enough independent factors for any set from [`arb_pz`].
///
/// All values lie in `[-1, 1]`, the domain of every factor.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Dependent factor values, positional by id; [`SAMPLE_LEN`] entries.
    pub alpha: Vec<f64>,
    /// Independent factor values, positional by generator; [`SAMPLE_LEN`] entries.
    pub beta: Vec<f64>,
}

/// Between 1 and 19 random [`Sample`]s.
pub fn arb_samples() -> impl Strategy<Value = Vec<Sample>> {
    vec(
        (
            vec(-1.0..=1.0f64, SAMPLE_LEN),
            vec(-1.0..=1.0f64, SAMPLE_LEN),
        )
            .prop_map(|(alpha, beta)| Sample { alpha, beta }),
        1..20,
    )
}

/// Assigns the sample's α values to `ids` in order, for lookups by [`FactorId`].
pub fn alpha_map(ids: &[FactorId], s: &Sample) -> HashMap<FactorId, f64> {
    ids.iter().zip(&s.alpha).map(|(&k, &a)| (k, a)).collect()
}

/// The point of `pz` at the sample's factor values.
///
/// Independent generators beyond the sample, such as the rounding-error
/// generators of the `sound` feature, get `β = 0`.
pub fn eval(pz: &SparsePolyZonotope, s: &Sample) -> DVector<f64> {
    let a = alpha_map(pz.ids(), s);

    pz.evaluate_with(
        |k| a[&k],
        &pad(&s.beta, pz.independent_generators().ncols()),
    )
}

/// The first `len` values of `coeffs`, with zeros appended if it is shorter.
///
/// Exactness checks feed the coefficients of the operands and leave the
/// generators the `sound` feature appends for rounding errors at zero; those
/// generators only widen the set by a few ulps, well within [`TOL`].
pub fn pad(coeffs: &[f64], len: usize) -> Vec<f64> {
    (0..len)
        .map(|j| coeffs.get(j).copied().unwrap_or(0.0))
        .collect()
}

/// Whether `a` and `b` agree up to [`TOL`], relative to the magnitude of `a`.
pub fn close(a: &DVector<f64>, b: &DVector<f64>) -> bool {
    (a - b).amax() <= TOL * (1.0 + a.amax())
}

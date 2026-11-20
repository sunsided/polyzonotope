//! The abstraction error of a linearization over a set of states and inputs.
//!
//! By Taylor's theorem with the Lagrange remainder, for `Δz = z − z*`,
//! `f(z) = f(z*) + J(z*) Δz + ½ Δzᵀ H(ξ) Δz` with `ξ` on the segment from `z*`
//! to `z`. With `J(z*) ∈ [A B] ± R_J` and `½ H(ξ) ∈ Q ± R_H` over a box that
//! holds every such segment, the error of the linear model is
//! `e = f(z) − (A x + B u + w) ∈ Δzᵀ Q Δz + [−ρ, ρ]`, with
//! `ρ = |Δz|ᵀ R_H |Δz| + R_J |Δz|`. The quadratic part stays a polynomial
//! zonotope, so its hull sees products of the same factor; only the radii go
//! through intervals.

use alloc::vec::Vec;

use nalgebra::DMatrix;
use polyzonotope::{Interval, IntervalBox, IntervalMatrix, SparsePolyZonotope, Zonotope};

use super::{Linearization, all_finite};
use crate::Dynamics;

/// An interval enclosure of the abstraction error over `states × u`, one
/// interval per state; `None` if a bound is not finite.
pub(super) fn abstraction_error<D: Dynamics>(
    dynamics: &D,
    model: &Linearization,
    states: &SparsePolyZonotope,
    u: &Zonotope,
) -> Option<Vec<Interval>> {
    let deviation = deviation(states, u, &model.point);
    let hull = deviation.interval_hull();
    if !all_finite(hull.0.iter()) {
        return None;
    }

    let hessians = dynamics.hessians(&segment_box(&hull, &model.point));
    let magnitude: Vec<Interval> = hull.0.iter().map(|iv| Interval::point(iv.mag())).collect();
    if !hessians.iter().all(|h| quadratic_fits(h, &magnitude)) {
        return None;
    }

    let (quadratic, radii): (Vec<_>, Vec<_>) = hessians
        .iter()
        .map(|h| h.scale(Interval::point(0.5)).center_radius())
        .unzip();
    let polynomial = deviation.quadratic_map(&quadratic).interval_hull();

    let error: Vec<Interval> = polynomial
        .0
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let rho = remainder_radius(&radii[i], &model.jacobian_radius, i, &magnitude);
            q + Interval { lo: -rho, hi: rho }
        })
        .collect();

    all_finite(error.iter()).then_some(error)
}

/// `Δz = (x − x*, u − u*)` over `states × u`, as one polynomial zonotope.
fn deviation(states: &SparsePolyZonotope, u: &Zonotope, point: &[f64]) -> SparsePolyZonotope {
    let n = states.dim();
    let shift = |v: &[f64]| -nalgebra::DVector::from_column_slice(v);
    let dx = states.translate(&shift(&point[..n]));
    if u.dim() == 0 {
        return dx;
    }

    dx.cartesian_product(&SparsePolyZonotope::from_zonotope(u).translate(&shift(&point[n..])))
}

/// A box that holds every segment from `z*` to a point of `z* + hull`: the
/// hull joined with zero, shifted by `z*`.
fn segment_box(hull: &IntervalBox, point: &[f64]) -> Vec<Interval> {
    hull.0
        .iter()
        .zip(point)
        .map(|(iv, &z)| Interval::new(iv.lo.min(0.0), iv.hi.max(0.0)) + Interval::point(z))
        .collect()
}

/// `ρ_i = |Δz|ᵀ R_H |Δz| + Σ_j R_J[i, j] |Δz_j|`, rounded up.
fn remainder_radius(
    hessian: &DMatrix<f64>,
    jacobian: &DMatrix<f64>,
    row: usize,
    magnitude: &[Interval],
) -> f64 {
    let quadratic = (0..magnitude.len()).fold(Interval::point(0.0), |acc, a| {
        (0..magnitude.len()).fold(acc, |acc, b| {
            acc + Interval::point(hessian[(a, b)]) * magnitude[a] * magnitude[b]
        })
    });
    let linear = (0..magnitude.len()).fold(quadratic, |acc, j| {
        acc + Interval::point(jacobian[(row, j)]) * magnitude[j]
    });

    linear.hi
}

/// Largest `|Δz|ᵀ |H| |Δz|` the error may reach: far beyond any useful
/// error, and far enough below `f64::MAX` that no later sum overflows,
/// which under `sound` would panic instead of giving an infinite bound.
const ERROR_LIMIT: f64 = 1e100;

/// Whether `|Δz|ᵀ |H| |Δz|`, which bounds every partial sum of the
/// quadratic map, stays below [`ERROR_LIMIT`]. Plain `f64` on purpose: an
/// overflow here gives infinity and a refusal, not a panic.
fn quadratic_fits(hessian: &IntervalMatrix, magnitude: &[Interval]) -> bool {
    let bound: f64 = (0..magnitude.len())
        .flat_map(|a| (0..magnitude.len()).map(move |b| (a, b)))
        .map(|(a, b)| hessian.0[(a, b)].mag() * magnitude[a].hi * magnitude[b].hi)
        .sum();

    bound < ERROR_LIMIT
}

//! Conservative polynomialization (Althoff 2013): the second-order part of
//! the abstraction error at the start of the step stays a polynomial
//! zonotope over the factors of the set, and only what changes during the
//! step goes through interval arithmetic.
//!
//! The error decomposition and the soundness argument are documented on
//! [`Method::Polynomialization`](crate::Method::Polynomialization), where
//! they show in the public docs.

use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{
    Interval, IntervalBox, IntervalMatrix, Reduction, SparsePolyZonotope, Zonotope,
};

use super::error::{deviation, segment_box};
use super::{Linearization, Term, Terms, points};
use crate::linear::without_factors;
use crate::{Dynamics, IntervalTensor, LinearSystem, ReachError};

/// Largest magnitude a box or derivative may have before the error is
/// computed: products of four such values and sums of a few thousand of
/// them stay far below `f64::MAX`, so no bound can overflow, which under
/// `sound` would panic instead of giving an infinite bound.
const TAME: f64 = 1e30;

/// The part of the abstraction error that is computed once per step, at the
/// start set `X`.
pub(super) struct Expansion {
    /// `½ H(z*)` per state, the Hessians at the linearization point.
    half_hessians: Vec<IntervalMatrix>,
    /// `Ψ_s = { sᵀ Q s }`, over the factors of `X`; `n` rows.
    pub(super) static_set: SparsePolyZonotope,
    /// `sᵀ R_Q s` bounded per state: the spread of `½ H(z*)` around `Q`,
    /// which only rounding makes nonzero.
    static_radius: Vec<f64>,
    /// The box of `s = (x_k − x_m, 0)`, one interval per entry of `z`.
    start: Vec<Interval>,
}

/// The static part of the error over `x`, with what [`dynamic_error`] needs
/// from the start set.
///
/// # Errors
///
/// - [`ReachError::MissingDerivatives`] if the dynamics supply no third
///   derivatives: the remainder would have no bound.
/// - [`ReachError::AbstractionDiverged`] if a bound is not finite or too
///   large to compute with.
pub(super) fn static_error<D: Dynamics>(
    dynamics: &D,
    model: &Linearization,
    x: &SparsePolyZonotope,
    (max_degree, order): (u32, f64),
) -> Result<Expansion, ReachError> {
    let at = points(model.point.iter().copied());
    if dynamics.third_derivatives(&at).is_none() {
        return Err(ReachError::MissingDerivatives);
    }

    let n = x.dim();
    let shifted = x.translate(&-DVector::from_column_slice(&model.middle));
    let hull = shifted.interval_hull();
    let half_hessians: Vec<IntervalMatrix> = dynamics
        .hessians(&at)
        .iter()
        .map(|h| h.scale(Interval::point(0.5)))
        .collect();
    let tame = tame(hull.0.iter()) && half_hessians.iter().all(|h| tame(h.0.iter()));
    if !tame {
        return Err(ReachError::AbstractionDiverged);
    }

    let (quadratic, radii): (Vec<_>, Vec<_>) = half_hessians
        .iter()
        .map(|h| {
            let (c, r) = h.center_radius();
            (c.view((0, 0), (n, n)).into_owned(), r)
        })
        .unzip();
    let static_set =
        cap_degree(&shifted.quadratic_map(&quadratic), max_degree).reduce(order, Reduction::Girard);
    let mut start = hull.0;
    start.resize(model.point.len(), Interval::point(0.0));

    Ok(Expansion {
        static_radius: radii.iter().map(|r| spread(r, &start)).collect(),
        half_hessians,
        static_set,
        start,
    })
}

/// `x' = A x + v + I e`, `v' = 0`: the linear model with the static error as
/// a constant input `v` in the lower `n` states, and the varying input `e`
/// (`B u + w + ψ_d`) on the upper ones.
pub(super) fn augmented_system(a: &DMatrix<f64>) -> LinearSystem {
    let n = a.nrows();
    let mut big = DMatrix::zeros(2 * n, 2 * n);
    big.view_mut((0, 0), (n, n)).copy_from(a);
    big.view_mut((0, n), (n, n))
        .copy_from(&DMatrix::identity(n, n));

    LinearSystem {
        a: big,
        b: DMatrix::identity(2 * n, n),
    }
}

/// The first `n` rows of `set`: the state part of an augmented set.
pub(super) fn project(set: &SparsePolyZonotope, n: usize) -> SparsePolyZonotope {
    set.linear_map(&DMatrix::identity(n, set.dim()))
}

/// An interval enclosure of the dynamic error `ψ_d` over the time-interval
/// set `states` and the inputs `u`, one interval per state, with the terms
/// `terms` selects (all of them, outside the tests).
///
/// # Errors
///
/// - [`ReachError::MissingDerivatives`] if the dynamics stop supplying third
///   derivatives.
/// - [`ReachError::AbstractionDiverged`] if a bound is not finite or too
///   large to compute with.
pub(super) fn dynamic_error<D: Dynamics>(
    dynamics: &D,
    (model, expansion): (&Linearization, &Expansion),
    (states, u): (&SparsePolyZonotope, &Zonotope),
    step: f64,
    terms: Terms,
) -> Result<Vec<Interval>, ReachError> {
    let boxes = Boxes::new(dynamics, model, expansion, (states, u), step)?;
    let mut deviation = boxes.deviation.clone();
    if !terms.has(Term::InputDeviation) {
        deviation[states.dim()..].fill(Interval::point(0.0));
    }

    let third = dynamics
        .third_derivatives(&segment_box(&IntervalBox(boxes.full.clone()), &model.point))
        .ok_or(ReachError::MissingDerivatives)?;
    if !third.iter().all(|t| tame(tensor_entries(t))) {
        return Err(ReachError::AbstractionDiverged);
    }

    let error = (0..states.dim())
        .map(|i| {
            let h = &expansion.half_hessians[i];
            let parts = [
                (Term::StaticRadius, symmetric(expansion.static_radius[i])),
                (Term::Cross, cross(h, &expansion.start, &deviation)),
                (Term::DynamicQuadratic, quadratic_form(h, &deviation)),
                (Term::Remainder, remainder(&third[i], &boxes.full)),
                (Term::JacobianSpread, jacobian_spread(model, i, &boxes.full)),
            ];
            parts
                .iter()
                .filter(|(term, _)| terms.has(*term))
                .fold(Interval::point(0.0), |acc, &(_, v)| acc + v)
        })
        .collect();

    Ok(error)
}

/// The boxes the dynamic error is computed from.
struct Boxes {
    /// `□Δz`: the hull of `R(τ) × U` minus `z*`.
    full: Vec<Interval>,
    /// `□δ`: `[0, r] · f(□R(τ) × □U) − (x* − x_m)` for the states,
    /// `□U − u*` for the inputs.
    deviation: Vec<Interval>,
}

impl Boxes {
    /// The boxes for the time-interval set `states` and the inputs `u`.
    ///
    /// # Errors
    ///
    /// [`ReachError::AbstractionDiverged`] if a box or the flow over it is
    /// not finite or too large to compute with.
    fn new<D: Dynamics>(
        dynamics: &D,
        model: &Linearization,
        expansion: &Expansion,
        (states, u): (&SparsePolyZonotope, &Zonotope),
        step: f64,
    ) -> Result<Self, ReachError> {
        let n = states.dim();
        let full = deviation(states, u, &model.point).interval_hull().0;
        if !tame(full.iter()) {
            return Err(ReachError::AbstractionDiverged);
        }

        let region: Vec<Interval> = full
            .iter()
            .zip(&model.point)
            .map(|(&d, &z)| d + Interval::point(z))
            .collect();
        let flow = dynamics.flow(&region);
        if !tame(flow.iter()) || !tame(expansion.start.iter()) {
            return Err(ReachError::AbstractionDiverged);
        }

        let duration = Interval::new(0.0, step);
        let deviation = flow
            .iter()
            .zip(model.point.iter().zip(&model.middle))
            .map(|(&f, (&star, &middle))| {
                duration * f - (Interval::point(star) - Interval::point(middle))
            })
            .chain(full[n..].iter().copied())
            .collect();

        Ok(Self { full, deviation })
    }
}

/// `sᵀ M δ + δᵀ M s` with interval products, `M = ½ H`.
fn cross(half_hessian: &IntervalMatrix, start: &[Interval], deviation: &[Interval]) -> Interval {
    pairs(start.len()).fold(Interval::point(0.0), |acc, (a, b)| {
        let m = half_hessian.0[(a, b)];
        acc + m * start[a] * deviation[b] + m * deviation[a] * start[b]
    })
}

/// `δᵀ M δ` with interval products, squares kept non-negative.
fn quadratic_form(half_hessian: &IntervalMatrix, deviation: &[Interval]) -> Interval {
    pairs(deviation.len()).fold(Interval::point(0.0), |acc, (a, b)| {
        let product = if a == b {
            square(deviation[a])
        } else {
            deviation[a] * deviation[b]
        };
        acc + half_hessian.0[(a, b)] * product
    })
}

/// `⅙ Σ_{abc} T_{abc} Δz_a Δz_b Δz_c`, grouped by the multiset `a ≤ b ≤ c`
/// so that repeated indices become powers: `Δz₀² Δz₁` stays on one side of
/// zero in `Δz₀`, which a product of three intervals would not see.
fn remainder(tensor: &IntervalTensor, full: &[Interval]) -> Interval {
    let w = full.len();
    let mut sum = Interval::point(0.0);
    for a in 0..w {
        for b in a..w {
            for c in b..w {
                sum = sum + coefficient(tensor, (a, b, c)) * monomial(full, (a, b, c));
            }
        }
    }

    sum / Interval::point(6.0)
}

/// The sum of the entries of `tensor` at every distinct ordering of the
/// multiset `(a, b, c)`, `a ≤ b ≤ c`: its coefficient in `Σ T Δz Δz Δz`.
fn coefficient(tensor: &IntervalTensor, (a, b, c): (usize, usize, usize)) -> Interval {
    let mut orderings = [
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a),
    ];
    orderings.sort_unstable();
    let mut distinct = orderings.to_vec();
    distinct.dedup();

    distinct
        .iter()
        .fold(Interval::point(0.0), |acc, &at| acc + tensor[at])
}

/// `Δz_a Δz_b Δz_c` for `a ≤ b ≤ c`, with squares where indices repeat.
fn monomial(full: &[Interval], (a, b, c): (usize, usize, usize)) -> Interval {
    match (a == b, b == c) {
        (true, true) => square(full[a]) * full[a],
        (true, false) => square(full[a]) * full[c],
        (false, true) => full[a] * square(full[b]),
        (false, false) => full[a] * full[b] * full[c],
    }
}

/// `R_J[i, :] |Δz|` as a symmetric interval: the spread of the Jacobian at
/// `z*` around `[A B]`, which only rounding makes nonzero.
fn jacobian_spread(model: &Linearization, row: usize, full: &[Interval]) -> Interval {
    let rho = full
        .iter()
        .enumerate()
        .fold(Interval::point(0.0), |acc, (j, d)| {
            acc + Interval::point(model.jacobian_radius[(row, j)]) * Interval::point(d.mag())
        });

    symmetric(rho.hi)
}

/// `|s|ᵀ R |s|` over the state block of `R`, rounded up.
fn spread(radius: &DMatrix<f64>, start: &[Interval]) -> f64 {
    pairs(start.len())
        .fold(Interval::point(0.0), |acc, (a, b)| {
            acc + Interval::point(radius[(a, b)])
                * Interval::point(start[a].mag())
                * Interval::point(start[b].mag())
        })
        .hi
}

/// Every index pair `(a, b)` of a `w × w` matrix.
fn pairs(w: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..w).flat_map(move |a| (0..w).map(move |b| (a, b)))
}

/// `{ x² | x ∈ iv }`: the interval product, cut at zero.
fn square(iv: Interval) -> Interval {
    let p = iv * iv;
    Interval::new(p.lo.max(0.0), p.hi)
}

/// `[−r, r]`.
fn symmetric(r: f64) -> Interval {
    Interval::new(-r, r)
}

/// Whether every bound is finite (and so not NaN) and at most [`TAME`] in
/// magnitude.
fn tame<'a>(mut intervals: impl Iterator<Item = &'a Interval>) -> bool {
    intervals.all(|iv| iv.lo.is_finite() && iv.hi.is_finite() && iv.mag() <= TAME)
}

/// The entries of `tensor`, for [`tame`].
fn tensor_entries(tensor: &IntervalTensor) -> impl Iterator<Item = &Interval> {
    let d = tensor.dim();
    (0..d * d * d).map(move |k| &tensor[(k / (d * d), (k / d) % d, k % d)])
}

/// `pz` with every dependent monomial of total degree above `max_degree`
/// enclosed in independent generators. Sound: the low monomials stay exact,
/// the high ones are replaced by their zonotope enclosure.
pub(super) fn cap_degree(pz: &SparsePolyZonotope, max_degree: u32) -> SparsePolyZonotope {
    let e = pz.exponents();
    let (low, high): (Vec<usize>, Vec<usize>) =
        (0..e.ncols()).partition(|&j| e.column(j).iter().sum::<u32>() <= max_degree);
    if high.is_empty() {
        return pz.clone();
    }

    let n = pz.dim();
    let part = |columns: &[usize], c: DVector<f64>, gi: DMatrix<f64>| {
        SparsePolyZonotope::new(
            c,
            pz.dependent_generators().select_columns(columns),
            gi,
            e.select_columns(columns),
            pz.ids().to_vec(),
        )
        .expect("columns of a valid set")
    };
    let kept = part(
        &low,
        pz.center().clone(),
        pz.independent_generators().clone(),
    );
    let enclosed = part(&high, DVector::zeros(n), DMatrix::zeros(n, 0)).to_zonotope();

    kept.exact_plus(&without_factors(&enclosed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coefficient of a multiset sums each distinct ordering once.
    #[test]
    fn coefficients_count_orderings_once() {
        let mut t = IntervalTensor::zeros(2);
        for (k, at) in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
            .into_iter()
            .enumerate()
        {
            t[at] = Interval::point(f64::from(u8::try_from(k + 1).unwrap()));
        }

        assert_eq!(coefficient(&t, (0, 0, 0)), Interval::point(1.0));
        assert_eq!(coefficient(&t, (0, 0, 1)), Interval::point(9.0));
        assert_eq!(coefficient(&t, (1, 1, 1)), Interval::point(5.0));
    }

    /// `⅙ T Δz³` for `x₀² x₁` (all six third derivatives with two zeros and
    /// one one equal 2) is `Δz₀² Δz₁`, and sees `Δz₀²` as non-negative.
    #[test]
    fn remainder_keeps_squares_non_negative() {
        let mut t = IntervalTensor::zeros(2);
        for at in [(0, 0, 1), (0, 1, 0), (1, 0, 0)] {
            t[at] = Interval::point(2.0);
        }

        let r = remainder(&t, &[Interval::new(-1.0, 1.0), Interval::new(1.0, 2.0)]);
        assert!(r.lo >= 0.0 && r.contains(2.0, 1e-12) && r.hi <= 2.0 + 1e-12);
    }

    /// Capping the degree keeps the low monomials and encloses the rest:
    /// `α + α³` capped at 1 still contains `α + α³` at sampled `α`.
    #[test]
    fn capped_set_contains_the_original() {
        let pz = SparsePolyZonotope::new(
            DVector::from_element(1, 0.5),
            DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            DMatrix::zeros(1, 0),
            DMatrix::from_row_slice(1, 2, &[1, 3]),
            polyzonotope::FactorId::fresh_n(1),
        )
        .unwrap();
        let capped = cap_degree(&pz, 1);
        assert!(capped.exponents().iter().all(|&e| e <= 1));
        assert_eq!(capped.ids(), pz.ids());

        for k in 0..=20 {
            let alpha = -1.0 + 0.1 * f64::from(k);
            let value = 0.5 + alpha + alpha * alpha * alpha;
            // The kept part at α plus the enclosure's β = α³ hits the value;
            // rounding generators, if any, stay at zero.
            let beta: Vec<f64> = (0..capped.independent_generators().ncols())
                .map(|k| if k == 0 { alpha * alpha * alpha } else { 0.0 })
                .collect();
            let at = capped.evaluate(&[alpha], &beta);
            assert!((at[0] - value).abs() < 1e-12, "α = {alpha}");
        }
    }

    /// The augmented system feeds `v` into `x` and keeps it constant.
    #[test]
    fn augmentation_layout() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let system = augmented_system(&a);
        let expected = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 2.0, 1.0, 0.0, //
                3.0, 4.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0,
            ],
        );
        assert_eq!(system.a, expected);
        assert_eq!(system.b, DMatrix::identity(4, 2));
    }
}

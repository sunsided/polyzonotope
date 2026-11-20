//! Reachable sets of linear systems `x' = A x + B u` with `u ∈ U`
//! (Althoff 2010, Sec. 3.2; Girard 2005 for the time-interval hull).

use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{Interval, IntervalMatrix, SparsePolyZonotope, Zonotope};

use crate::ReachError;
use crate::taylor::Taylor;

/// `x' = A x + B u`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSystem {
    /// State matrix `A`, `n × n`.
    pub a: DMatrix<f64>,
    /// Input matrix `B`, `n × m`.
    pub b: DMatrix<f64>,
}

/// The reachable sets of one step `[t_k, t_k + r]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// Every state the system can be in at `t_k + r`.
    pub time_point: SparsePolyZonotope,
    /// Every state the system can be in at some `t ∈ [t_k, t_k + r]`.
    pub time_interval: SparsePolyZonotope,
}

/// Precomputed one-step propagation of a [`LinearSystem`] with time step `r`.
pub struct LinearReach {
    /// The system.
    system: LinearSystem,
    /// Taylor series of `e^{A r}` with its remainder.
    taylor: Taylor,
    /// `e^{A r}`, enclosed.
    exponential: IntervalMatrix,
    /// `Γ(r)`, the response to a constant input after one step.
    input_solution: IntervalMatrix,
    /// `F`, the time-interval correction of the state.
    state_correction: IntervalMatrix,
    /// `F̃`, the time-interval correction of a constant input.
    input_correction: IntervalMatrix,
}

impl LinearReach {
    /// Propagation of `system` in steps of `step`, with `taylor_terms` terms
    /// of the matrix exponential after the first.
    ///
    /// # Errors
    ///
    /// [`ReachError::StepTooLarge`] if the Taylor remainder bound does not
    /// hold for this step and term count.
    ///
    /// # Panics
    ///
    /// Panics if `A` is not square or `B` does not have one row per state.
    pub fn new(system: LinearSystem, step: f64, taylor_terms: usize) -> Result<Self, ReachError> {
        let n = system.a.nrows();
        assert_eq!(system.a.ncols(), n, "A must be square");
        assert_eq!(system.b.nrows(), n, "B needs one row per state");
        let taylor = Taylor::new(&system.a, step, taylor_terms)?;

        Ok(Self {
            exponential: taylor.exponential(),
            input_solution: taylor.input_solution(),
            state_correction: taylor.state_correction(),
            input_correction: taylor.input_correction(),
            taylor,
            system,
        })
    }

    /// The reachable sets one step after `x`, for inputs in `u`: any
    /// integrable signal with values in `u`, the zonotope's center being the
    /// constant part.
    ///
    /// Time point: `e^{Ar} X ⊕ Γ(r) B c_u ⊕ R_p(B U_Δ)`. Time interval: the
    /// hull of `X` and `e^{Ar} X + Γ(r) B c_u`, plus `F X ⊕ F̃ B c_u ⊕ R_p(B U_Δ)`, where
    /// `R_p(V) = ⊕_{i=0}^{η} A^i r^{i+1} / (i+1)! V ⊕ E(r) r V` encloses the
    /// response to the varying part `U_Δ` of the input.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have one row per state or `u` one per input.
    #[must_use]
    pub fn step(&self, x: &SparsePolyZonotope, u: &Zonotope) -> Step {
        let constant = &self.system.b * u.center();
        let varying = Zonotope::new(DVector::zeros(u.dim()), u.generators().clone())
            .linear_map(&self.system.b);
        let particular = self.varying_input_response(&varying);

        let input_effect = point_response(&self.input_solution, &constant);
        let time_point = x
            .interval_map(&self.exponential)
            .minkowski_sum(&input_effect)
            .minkowski_sum(&particular);
        let time_interval = hull_with_successor(x, &self.exponential, &input_effect)
            .minkowski_sum(&x.interval_map(&self.state_correction))
            .minkowski_sum(&point_response(&self.input_correction, &constant))
            .minkowski_sum(&particular);

        Step {
            time_point,
            time_interval,
        }
    }

    /// `R_p(V) = ⊕_{i=0}^{η} A^i r^{i+1} / (i+1)! V ⊕ E(r) r V`.
    fn varying_input_response(&self, v: &Zonotope) -> SparsePolyZonotope {
        let pz = without_factors(v);
        (0..=self.taylor.terms())
            .map(|i| pz.interval_map(&self.taylor.input_term(i)))
            .fold(
                pz.interval_map(
                    &self
                        .taylor
                        .remainder
                        .scale(Interval::point(self.taylor.step)),
                ),
                |acc, term| acc.minkowski_sum(&term),
            )
    }
}

/// `{ M v | M ∈ m }` for a fixed vector `v`, as a set without factors.
fn point_response(m: &IntervalMatrix, v: &DVector<f64>) -> SparsePolyZonotope {
    let point = Zonotope::new(v.clone(), DMatrix::zeros(v.len(), 0));
    without_factors(&point.interval_map(m))
}

/// A zonotope as a polynomial zonotope with independent generators only, so
/// that its parts never correlate with anything else.
fn without_factors(z: &Zonotope) -> SparsePolyZonotope {
    SparsePolyZonotope::new(
        z.center().clone(),
        DMatrix::zeros(z.dim(), 0),
        z.generators().clone(),
        DMatrix::zeros(0, 0),
        Vec::new(),
    )
    .expect("consistent shapes")
}

/// An enclosure of `{ λ x + (1 − λ) (e^{Ar} x + w) | x ∈ X, λ ∈ [0, 1] }`
/// (Girard 2005), with `Φ = Φ_c ± R` the interval matrix `phi` and `w` from
/// `offset`, a point `w_c` plus independent generators:
///
/// `λ x + (1 − λ)(Φ_c x + w_c) = ½ (x + Φ_c x + w_c) + (λ − ½)(x − Φ_c x − w_c)`.
/// The first part keeps the factors of `X`. In the second, `2λ − 1 ∈ [-1, 1]`
/// multiplies every term, so each generator of `½ (X − Φ_c X − w_c)`, its
/// center included, becomes an independent generator: even monomials lose
/// their sign there. `(Φ − Φ_c) x ∈ [-R, R] X` covers the spread of `phi`,
/// and the generators of `offset` cover the spread of `w`.
fn hull_with_successor(
    x: &SparsePolyZonotope,
    phi: &IntervalMatrix,
    offset: &SparsePolyZonotope,
) -> SparsePolyZonotope {
    let n = x.dim();
    let (center, radius) = phi.center_radius();
    let half = DMatrix::from_diagonal_element(n, n, 0.5);
    let successor = x.linear_map(&center).translate(offset.center());

    let mean = x.linear_map(&half).exact_plus(&successor.linear_map(&half));
    let difference = x
        .linear_map(&half)
        .exact_plus(&successor.linear_map(&-&half));
    let swing = SparsePolyZonotope::new(
        DVector::zeros(n),
        DMatrix::zeros(n, 0),
        columns(&[
            difference.dependent_generators(),
            difference.independent_generators(),
            &DMatrix::from_column_slice(n, 1, difference.center().as_slice()),
        ]),
        DMatrix::zeros(0, 0),
        Vec::new(),
    )
    .expect("consistent shapes");
    let spread = IntervalMatrix(radius.map(|r| Interval::new(-r, r)));
    let offset_spread = offset.translate(&-offset.center());

    mean.minkowski_sum(&swing)
        .minkowski_sum(&x.interval_map(&spread))
        .minkowski_sum(&offset_spread)
}

/// The columns of all `parts`, side by side; all have the same row count.
fn columns(parts: &[&DMatrix<f64>]) -> DMatrix<f64> {
    let rows = parts[0].nrows();
    let all: Vec<f64> = parts.iter().flat_map(|m| m.iter().copied()).collect();
    DMatrix::from_vec(rows, all.len() / rows.max(1), all)
}

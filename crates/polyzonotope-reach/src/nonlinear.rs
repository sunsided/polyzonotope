//! Reachable sets of nonlinear systems `x' = f(x, u)` by conservative
//! linearization (Althoff, Stursberg & Buss 2008) or conservative
//! polynomialization (Althoff 2013), with polynomial zonotopes (Kochdumper &
//! Althoff 2021).
//!
//! Each step linearizes `f` around a point `z* = (x*, u*)` and treats the
//! rest, the abstraction error, as an extra uncertain input of the linear
//! system. The error depends on the time-interval set it is meant to
//! produce, so it is guessed, checked and enlarged until the check passes.
//! With [`Method::Polynomialization`], the second-order error at the start
//! set is computed once, as a polynomial zonotope over the set's factors,
//! and fed in exactly; only the rest is guessed:
//!
//! ```mermaid
//! flowchart LR
//!     L[linearize at z*] --> P[static error Ψ_s over X, polynomialization only]
//!     P --> R[linear step with error box Ψ̄]
//!     R --> E[error Ψ over the time-interval set]
//!     E -->|Ψ ⊆ Ψ̄| A[accept, reduce]
//!     E -->|Ψ ⊄ Ψ̄| W[Ψ̄ := widen Ψ̄ ∪ Ψ]
//!     W --> R
//!     W -->|too many tries| S[split the set or refuse]
//! ```

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{Interval, IntervalBox, Reduction, SparsePolyZonotope, Zonotope};

use crate::{Dynamics, LinearReach, LinearSystem, ReachError, ReachFailure, Step};

mod error;
mod polynomialization;

#[cfg(test)]
mod ablation;

/// How [`NonlinearReach`] abstracts the dynamics in each step.
///
/// Non-exhaustive: more abstractions may be added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// Conservative linearization (Althoff, Stursberg & Buss 2008): the
    /// whole abstraction error `½ Δzᵀ H(ξ) Δz` over the time-interval set is
    /// one box, guessed and checked. Needs the Hessians only. The default,
    /// because every [`Dynamics`] supports it.
    #[default]
    Linearization,
    /// Conservative polynomialization (Althoff 2013): the second-order error
    /// at the start set stays a polynomial zonotope over the set's factors
    /// and moves with the set; only what changes during the step and a
    /// third-order Lagrange remainder go into the box. Needs
    /// [`Dynamics::third_derivatives`]. Tighter for large sets, since the box
    /// shrinks from `O(w²)` to `O(w r + w³)` for a set of width `w`.
    Polynomialization,
}

/// How [`NonlinearReach`] steps through time.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearSettings {
    /// Time step `r`, positive.
    pub step: f64,
    /// Terms of the matrix exponential after the first, see [`LinearReach::new`].
    pub taylor_terms: usize,
    /// Order (generators per dimension, at least 1) the sets are reduced to
    /// after every step and before the error is computed.
    pub order: f64,
    /// How often the error guess may be enlarged before the step is given up.
    pub max_iterations: usize,
    /// Relative widening of the error guess on each retry: every bound moves
    /// outward by this fraction of the interval's magnitude.
    pub enlarge: f64,
    /// Most pieces a set may be split into per time step before a failing
    /// step is refused; 1 never splits.
    pub max_pieces: usize,
    /// The abstraction of each step.
    pub method: Method,
    /// Highest total degree the static error of
    /// [`Method::Polynomialization`] keeps exact, at least 1. It is
    /// quadratic in the set, so without a cap the degree of the sets would
    /// double with every step; higher monomials are enclosed in independent
    /// generators.
    pub max_degree: u32,
}

impl NonlinearSettings {
    /// Defaults for time step `step`: 4 Taylor terms, order 20, 10
    /// iterations, 10% widening, up to 8 pieces, linearization, and degree
    /// 4 for polynomialization.
    #[must_use]
    pub fn new(step: f64) -> Self {
        Self {
            step,
            taylor_terms: 4,
            order: 20.0,
            max_iterations: 10,
            enlarge: 0.1,
            max_pieces: 8,
            method: Method::Linearization,
            max_degree: 4,
        }
    }
}

/// One-step and multi-step reachability of [`Dynamics`].
///
/// # Examples
///
/// ```
/// use nalgebra::{DMatrix, DVector};
/// use polyzonotope_reach::polyzonotope::{SparsePolyZonotope, Zonotope};
/// use polyzonotope_reach::{NonlinearReach, NonlinearSettings, Polynomial};
///
/// // Van der Pol with μ = 1, no inputs.
/// let vdp = Polynomial::autonomous(2)
///     .term(0, 1.0, &[0, 1])
///     .term(1, 1.0, &[0, 1])
///     .term(1, -1.0, &[2, 1])
///     .term(1, -1.0, &[1, 0]);
/// let reach = NonlinearReach::new(vdp, NonlinearSettings::new(0.02));
/// let start = SparsePolyZonotope::from_zonotope(&Zonotope::new(
///     DVector::from_vec(vec![1.4, 2.4]),
///     DMatrix::from_diagonal_element(2, 2, 0.05),
/// ));
///
/// let flowpipe = reach.reach_autonomous(&start, 10)?;
/// let hull = flowpipe[9][0].time_point.interval_hull();
/// assert!(hull.0[0].lo > 1.4 && hull.0[1].hi < 2.4);
/// # Ok::<(), polyzonotope_reach::ReachFailure>(())
/// ```
pub struct NonlinearReach<D> {
    /// The vector field.
    dynamics: D,
    /// Step size and limits, checked by [`new`](Self::new).
    settings: NonlinearSettings,
}

/// The linear model `x' = A x + B u + w` of one step, with
/// `w ∈ f(z*) − A x* − B u*`.
pub(crate) struct Linearization {
    /// The point `z* = (x*, u*)`.
    point: Vec<f64>,
    /// `x_m`, the middle of the interval hull of the start set, from which
    /// `x*` is half a step along `f`; `n` entries.
    middle: Vec<f64>,
    /// `A`, the center of `∂f/∂x` at `z*`.
    a: DMatrix<f64>,
    /// `B`, the center of `∂f/∂u` at `z*`.
    b: DMatrix<f64>,
    /// An enclosure of `f(z*) − A x* − B u*`.
    offset: Vec<Interval>,
    /// The radius of `∂f/∂z` at `z*` around `[A B]`: zero unless rounding
    /// made the point evaluation an interval.
    jacobian_radius: DMatrix<f64>,
}

impl<D: Dynamics> NonlinearReach<D> {
    /// Reachability of `dynamics` with `settings`.
    ///
    /// # Panics
    ///
    /// Panics if the step is not positive and finite, the order is below 1,
    /// `enlarge` is negative, or `max_pieces` or `max_degree` is zero.
    #[must_use]
    pub fn new(dynamics: D, settings: NonlinearSettings) -> Self {
        assert!(
            settings.step > 0.0 && settings.step.is_finite(),
            "step must be positive"
        );
        assert!(settings.order >= 1.0, "order must be at least 1");
        assert!(settings.enlarge >= 0.0, "enlarge must not be negative");
        assert!(settings.max_pieces >= 1, "max_pieces must be at least 1");
        assert!(settings.max_degree >= 1, "max_degree must be at least 1");

        Self { dynamics, settings }
    }

    /// The reachable sets one step after `x`, for any integrable input
    /// signal with values in `u`; for an autonomous system, `u` is the
    /// zero-dimensional zonotope (or use
    /// [`step_autonomous`](Self::step_autonomous)).
    ///
    /// Both sets are reduced to the configured order, so the time-point set
    /// can go straight into the next step.
    ///
    /// # Errors
    ///
    /// - [`ReachError::StepTooLarge`] if the matrix exponential's remainder
    ///   bound fails for the linearized `A`.
    /// - [`ReachError::AbstractionDiverged`] if no error guess was confirmed
    ///   within `max_iterations`, or the model or the error stopped being
    ///   finite: the set or the step is too large for the abstraction. No
    ///   set is returned then, since none is proven.
    /// - [`ReachError::MissingDerivatives`] with
    ///   [`Method::Polynomialization`] if the dynamics supply no third
    ///   derivatives.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have one row per state or `u` one per input.
    pub fn step(&self, x: &SparsePolyZonotope, u: &Zonotope) -> Result<Step, ReachError> {
        self.step_with(x, u, Terms::ALL)
    }

    /// [`step`](Self::step) with only the error terms in `terms`; the tests
    /// drop one at a time to show that each is needed.
    fn step_with(
        &self,
        x: &SparsePolyZonotope,
        u: &Zonotope,
        terms: Terms,
    ) -> Result<Step, ReachError> {
        assert_eq!(x.dim(), self.dynamics.states(), "x needs one row per state");
        assert_eq!(u.dim(), self.dynamics.inputs(), "u needs one row per input");

        let model = self
            .linearize(x, u)
            .ok_or(ReachError::AbstractionDiverged)?;
        match self.settings.method {
            Method::Linearization => self.step_linearized(x, u, &model, terms),
            Method::Polynomialization => self.step_polynomialized(x, u, &model, terms),
        }
    }

    /// One step of [`Method::Linearization`]: the linear model with the
    /// whole error as a guessed box.
    fn step_linearized(
        &self,
        x: &SparsePolyZonotope,
        u: &Zonotope,
        model: &Linearization,
        terms: Terms,
    ) -> Result<Step, ReachError> {
        let n = self.dynamics.states();
        let system = LinearSystem {
            a: model.a.clone(),
            b: DMatrix::identity(n, n),
        };
        let linear = LinearReach::new(system, self.settings.step, self.settings.taylor_terms)?;
        let known = u.linear_map(&model.b);

        self.confirm(n, |assumed| {
            let step = linear.step(x, &with_error(&known, &model.offset, assumed));
            let states = step
                .time_interval
                .reduce(self.settings.order, Reduction::Girard);
            let actual = error::abstraction_error(&self.dynamics, model, (&states, u), terms)
                .ok_or(ReachError::AbstractionDiverged)?;

            Ok((step.time_point, states, actual))
        })
    }

    /// One step of [`Method::Polynomialization`]: the static error `Ψ_s`
    /// rides along as the constant input of an augmented linear system, and
    /// only the dynamic error is a guessed box. The soundness argument is in
    /// the `polynomialization` module.
    fn step_polynomialized(
        &self,
        x: &SparsePolyZonotope,
        u: &Zonotope,
        model: &Linearization,
        terms: Terms,
    ) -> Result<Step, ReachError> {
        let n = self.dynamics.states();
        let limits = (self.settings.max_degree, self.settings.order);
        let expansion = polynomialization::static_error(&self.dynamics, model, x, limits)?;
        let start = if terms.has(Term::StaticSet) {
            x.cartesian_product(&expansion.static_set)
        } else {
            x.cartesian_product(&SparsePolyZonotope::from_zonotope(&Zonotope::new(
                DVector::zeros(n),
                DMatrix::zeros(n, 0),
            )))
        };
        let system = polynomialization::augmented_system(&model.a);
        let linear = LinearReach::new(system, self.settings.step, self.settings.taylor_terms)?;
        let known = u.linear_map(&model.b);

        self.confirm(n, |assumed| {
            let step = linear.step(&start, &with_error(&known, &model.offset, assumed));
            let states = polynomialization::project(&step.time_interval, n)
                .reduce(self.settings.order, Reduction::Girard);
            let actual = polynomialization::dynamic_error(
                &self.dynamics,
                (model, &expansion),
                (&states, u),
                self.settings.step,
                terms,
            )?;

            Ok((
                polynomialization::project(&step.time_point, n),
                states,
                actual,
            ))
        })
    }

    /// The guess-and-check loop: `attempt` runs the step with the error
    /// guess `Ψ̄` and returns the time-point set, the reduced time-interval
    /// set and the error over it. The first attempt whose error lies in its
    /// guess is accepted; otherwise the guess widens, up to
    /// `max_iterations` times.
    ///
    /// # Errors
    ///
    /// The first error of `attempt`, or [`ReachError::AbstractionDiverged`]
    /// if no guess was confirmed or the guess stopped being finite.
    fn confirm(
        &self,
        n: usize,
        mut attempt: impl FnMut(
            &[Interval],
        ) -> Result<
            (SparsePolyZonotope, SparsePolyZonotope, Vec<Interval>),
            ReachError,
        >,
    ) -> Result<Step, ReachError> {
        let mut assumed = vec![Interval::point(0.0); n];
        for _ in 0..self.settings.max_iterations {
            let (time_point, time_interval, actual) = attempt(&assumed)?;
            if encloses(&assumed, &actual) {
                return Ok(Step {
                    time_point: time_point.reduce(self.settings.order, Reduction::Girard),
                    time_interval,
                });
            }

            assumed = widen(&assumed, &actual, self.settings.enlarge);
            if !all_finite(assumed.iter()) {
                break;
            }
        }

        Err(ReachError::AbstractionDiverged)
    }

    /// `steps` consecutive steps from `x0`: entry `k` holds the pieces of
    /// step `k`, whose union encloses the reachable set over
    /// `[k r, (k + 1) r]`.
    ///
    /// A piece whose step diverges is enclosed in an oriented box, split in
    /// two along its longest axis and retried, as long as the step stays
    /// within `max_pieces`.
    /// The pieces stay separate for the rest of the horizon.
    ///
    /// # Errors
    ///
    /// A [`ReachFailure`] with the first error of [`step`](Self::step) that
    /// splitting could not avoid, and every step proven before it.
    ///
    /// # Panics
    ///
    /// Panics if `x0` does not have one row per state or `u` one per input.
    pub fn reach(
        &self,
        x0: &SparsePolyZonotope,
        u: &Zonotope,
        steps: usize,
    ) -> Result<Vec<Vec<Step>>, ReachFailure> {
        let mut pieces = vec![x0.clone()];
        let mut flowpipe = Vec::with_capacity(steps);
        for _ in 0..steps {
            let next = match self.advance(pieces, u) {
                Ok(next) => next,
                Err(error) => {
                    return Err(ReachFailure {
                        proven: flowpipe,
                        error,
                    });
                }
            };
            pieces = next.iter().map(|s| s.time_point.clone()).collect();
            flowpipe.push(next);
        }

        Ok(flowpipe)
    }

    /// [`step`](Self::step) for an autonomous system `x' = f(x)`, without
    /// building the zero-dimensional input set.
    ///
    /// # Errors
    ///
    /// As for [`step`](Self::step).
    ///
    /// # Panics
    ///
    /// Panics if the dynamics have inputs, or `x` does not have one row per
    /// state.
    pub fn step_autonomous(&self, x: &SparsePolyZonotope) -> Result<Step, ReachError> {
        self.step(x, &self.no_input())
    }

    /// [`reach`](Self::reach) for an autonomous system `x' = f(x)`, without
    /// building the zero-dimensional input set.
    ///
    /// # Errors
    ///
    /// As for [`reach`](Self::reach).
    ///
    /// # Panics
    ///
    /// Panics if the dynamics have inputs, or `x0` does not have one row per
    /// state.
    pub fn reach_autonomous(
        &self,
        x0: &SparsePolyZonotope,
        steps: usize,
    ) -> Result<Vec<Vec<Step>>, ReachFailure> {
        self.reach(x0, &self.no_input(), steps)
    }

    /// The zero-dimensional input set of an autonomous system.
    ///
    /// # Panics
    ///
    /// Panics if the dynamics have inputs: dropping them silently would
    /// compute the sets for `u = 0` only.
    fn no_input(&self) -> Zonotope {
        assert_eq!(
            self.dynamics.inputs(),
            0,
            "an autonomous step needs dynamics without inputs"
        );
        Zonotope::new(DVector::zeros(0), DMatrix::zeros(0, 0))
    }

    /// One step for every piece in `todo`, splitting pieces that diverge.
    fn advance(
        &self,
        mut todo: Vec<SparsePolyZonotope>,
        u: &Zonotope,
    ) -> Result<Vec<Step>, ReachError> {
        let mut done = Vec::with_capacity(todo.len());
        while let Some(piece) = todo.pop() {
            match self.step(&piece, u) {
                Ok(step) => done.push(step),
                Err(ReachError::AbstractionDiverged)
                    if done.len() + todo.len() + 2 <= self.settings.max_pieces =>
                {
                    let halves = halves(&piece);
                    if halves.len() < 2 {
                        return Err(ReachError::AbstractionDiverged);
                    }

                    todo.extend(halves);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(done)
    }

    /// The linear model around `z* = (x* , c_u)`, where `x*` is the middle
    /// of `x`'s interval hull moved half a step along `f` (as CORA does);
    /// `None` if a value along the way is not finite.
    fn linearize(&self, x: &SparsePolyZonotope, u: &Zonotope) -> Option<Linearization> {
        let n = self.dynamics.states();
        let middle: Vec<f64> = x
            .interval_hull()
            .0
            .iter()
            .map(|iv| f64::midpoint(iv.lo, iv.hi))
            .collect();
        let inputs = u.center().iter().copied();
        let drift = self
            .dynamics
            .flow(&points(middle.iter().copied().chain(inputs.clone())));
        let point: Vec<f64> = middle
            .iter()
            .zip(&drift)
            .map(|(&m, d)| m + 0.5 * self.settings.step * f64::midpoint(d.lo, d.hi))
            .chain(inputs)
            .collect();
        if !point.iter().all(|v| v.is_finite()) {
            return None;
        }

        let at = points(point.iter().copied());
        let jacobian = self.dynamics.jacobian(&at);
        let flow = self.dynamics.flow(&at);
        if !all_finite(jacobian.0.iter()) || !all_finite(flow.iter()) {
            return None;
        }

        let (jacobian, jacobian_radius) = jacobian.center_radius();
        let offset = model_offset(&flow, &jacobian, &point);
        if !all_finite(offset.iter()) {
            return None;
        }

        Some(Linearization {
            a: jacobian.columns(0, n).into_owned(),
            b: jacobian.columns(n, point.len() - n).into_owned(),
            point,
            middle,
            offset,
            jacobian_radius,
        })
    }
}

/// One term of the abstraction error, so that the tests can drop it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Term {
    /// Linearization: `Δzᵀ Q Δz` over the time-interval set.
    Quadratic,
    /// Linearization: the spread of `½ H` over the segment box.
    HessianSpread,
    /// Both: the spread of the Jacobian at `z*`, from rounding only.
    JacobianSpread,
    /// Polynomialization: the static polynomial `Ψ_s = sᵀ Q s`.
    StaticSet,
    /// Polynomialization: the spread of `½ H(z*)` around `Q` over `s`, from
    /// rounding only.
    StaticRadius,
    /// Polynomialization: `sᵀ (½H) δ + δᵀ (½H) s`.
    Cross,
    /// Polynomialization: `δᵀ (½H) δ`.
    DynamicQuadratic,
    /// Polynomialization: the third-order Lagrange remainder.
    Remainder,
    /// Polynomialization: the input part `u − u*` of `δ`.
    InputDeviation,
}

/// A set of [`Term`]s, as a bit mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Terms(u16);

impl Terms {
    /// Every term: what the public API always uses.
    pub(crate) const ALL: Self = Self(u16::MAX);

    /// Whether `term` is in the set.
    pub(crate) fn has(self, term: Term) -> bool {
        self.0 & Self::bit(term) != 0
    }

    /// The set without `term`.
    #[cfg(test)]
    pub(crate) fn without(self, term: Term) -> Self {
        Self(self.0 & !Self::bit(term))
    }

    /// The bit of `term`.
    fn bit(term: Term) -> u16 {
        1 << term as u16
    }
}

/// Whether every bound in `intervals` is finite (and so not NaN either).
fn all_finite<'a>(mut intervals: impl Iterator<Item = &'a Interval>) -> bool {
    intervals.all(|iv| iv.lo.is_finite() && iv.hi.is_finite())
}

/// `piece` in two: its oriented box (PCA reduction to order 1), halved
/// along the longest axis. Splitting one factor of a set with many barely
/// shrinks it; the box loses the dependencies but halves the width.
fn halves(piece: &SparsePolyZonotope) -> Vec<SparsePolyZonotope> {
    let oriented = piece.reduce(1.0, Reduction::Pca).to_zonotope();
    SparsePolyZonotope::from_zonotope(&oriented).split_pieces(1)
}

/// Point intervals for `values`.
fn points(values: impl Iterator<Item = f64>) -> Vec<Interval> {
    values.map(Interval::point).collect()
}

/// `f(z*) − J z*` row by row in interval arithmetic, `J = [A B]`.
fn model_offset(flow: &[Interval], jacobian: &DMatrix<f64>, point: &[f64]) -> Vec<Interval> {
    flow.iter()
        .enumerate()
        .map(|(i, &f)| {
            point.iter().enumerate().fold(f, |acc, (j, &z)| {
                acc - Interval::point(jacobian[(i, j)]) * Interval::point(z)
            })
        })
        .collect()
}

/// The input of the linear model: `B U ⊕ (w + Ψ̄)` with the box around its
/// midpoint, so its center is the constant part.
fn with_error(known: &Zonotope, offset: &[Interval], assumed: &[Interval]) -> Zonotope {
    let total = offset.iter().zip(assumed).map(|(&w, &e)| w + e).collect();
    known.minkowski_sum(&Zonotope::from_box(&IntervalBox(total)))
}

/// Whether every interval of `actual` lies in the matching one of `assumed`.
fn encloses(assumed: &[Interval], actual: &[Interval]) -> bool {
    assumed
        .iter()
        .zip(actual)
        .all(|(a, e)| a.lo <= e.lo && e.hi <= a.hi)
}

/// The hull of `assumed` and `actual`, every bound moved outward by
/// `enlarge` times the interval's magnitude.
fn widen(assumed: &[Interval], actual: &[Interval], enlarge: f64) -> Vec<Interval> {
    assumed
        .iter()
        .zip(actual)
        .map(|(a, e)| {
            let hull = Interval::new(a.lo.min(e.lo), a.hi.max(e.hi));
            let margin = enlarge * hull.mag();
            Interval::new(hull.lo - margin, hull.hi + margin)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Widening covers both boxes and moves out by the relative margin.
    #[test]
    fn widening_covers_both() {
        let w = widen(&[Interval::new(-1.0, 1.0)], &[Interval::new(0.0, 2.0)], 0.5);
        assert_eq!(w, vec![Interval::new(-2.0, 3.0)]);
        assert!(encloses(&w, &[Interval::new(-1.0, 2.0)]));
        assert!(!encloses(&w, &[Interval::new(-1.0, 3.5)]));
    }
}

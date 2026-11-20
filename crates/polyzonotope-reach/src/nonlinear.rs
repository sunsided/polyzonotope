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

use crate::{Dynamics, LinearReach, LinearSystem, ReachError, Step};

mod driver;
mod error;
mod polynomialization;
mod split;

pub use driver::Adaptive;

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
    ///
    /// # The error decomposition
    ///
    /// Let `f` be three times continuously differentiable, `z* = (x*, u*)` the
    /// linearization point and `[A B]` the center of the enclosure of `J(z*)`,
    /// with `w ∈ f(z*) − A x* − B u*` the model offset. For a trajectory with
    /// `x(t_k) = x_k` and any input signal `u(t) ∈ U`, write
    ///
    /// - `Δz(t) = z(t) − z*` with `z(t) = (x(t), u(t))`,
    /// - `s = (x_k − x_m, 0)`, the start of the trajectory relative to the
    ///   middle `x_m` of the start set's hull, which is constant along the
    ///   trajectory, and
    /// - `δ(t) = Δz(t) − s = (x(t) − x_k − c, u(t) − u*)`, what changes, with
    ///   `c = x* − x_m` the half step from `x_m` to the linearization point.
    ///
    /// Any constant split of `Δz` would do; this one centers both parts. With
    /// `s = (x_k − x*, 0)` instead, `s` would be off-center by the half step and
    /// `δ` would range over the whole step's drift `[0, r] f`, which made the
    /// sets of small starts larger than those of linearization in a trial on
    /// Van der Pol. Centered, `δ` spans about `±½ r f`.
    ///
    /// Taylor's theorem with the Lagrange remainder gives, row by row and for
    /// some `ξ(t)` on the segment from `z*` to `z(t)`,
    ///
    /// ```text
    /// f(z) = f(z*) + J(z*) Δz + ½ Δzᵀ H(z*) Δz + ⅙ Σ_{abc} ∂³f(ξ)_{abc} Δz_a Δz_b Δz_c.
    /// ```
    ///
    /// With `½ H(z*) ∈ Q ± R_Q` and `J(z*) ∈ [A B] ± R_J`, the error of the
    /// linear model, `e = f(z) − (A x + B u + w)`, splits into
    ///
    /// ```text
    /// e = ψ_s(x_k) + ψ_d(t),
    /// ψ_s(x_k) = sᵀ Q s                                          (static)
    /// ψ_d(t)  ∈ sᵀ [±R_Q] s + sᵀ (½H) δ + δᵀ (½H) s + δᵀ (½H) δ  (dynamic)
    ///           + ⅙ Σ T(ξ) Δz Δz Δz + [±R_J] |Δz|.
    /// ```
    ///
    /// The inputs are never part of `s`: `u(t)` may jump anywhere in `U` during
    /// the step, so only the state at `t_k` is constant along a trajectory.
    /// Every term with `u − u*` is dynamic.
    ///
    /// # Why the sets enclose every trajectory
    ///
    /// The static part is a polynomial of `x_k` alone. The step computes
    /// `Ψ_s = { sᵀ Q s | x_k ∈ X }` with the quadratic map of the polynomial
    /// zonotope `X − x_m`, which keeps the factors of `X`: the point of `Ψ_s` that
    /// belongs to `x_k = X(α, β)` uses the same `α`. Degrees above the cap are
    /// enclosed as independent generators, which only adds points.
    ///
    /// The static error acts on the trajectory as a constant input `v = ψ_s(x_k)`.
    /// The step therefore runs the linear reach on the augmented system
    /// `x' = A x + v + B u + w + ψ_d`, `v' = 0`, from
    /// `X × Ψ_s` built with [`SparsePolyZonotope::cartesian_product`](polyzonotope::SparsePolyZonotope::cartesian_product), which
    /// keeps the shared factors: `(x_k, ψ_s(x_k))` is a point of it. The linear
    /// reach is sound for any input signal in its input set, so with
    /// `ψ_d(t) ∈ Ψ̄` for all `t` of the step, the projection of its sets onto
    /// `x` holds the trajectory at `t_{k+1}` and during the step. The static
    /// error never needs a guess: it is exact up to the degree cap, and it moves
    /// with the set rather than being smeared over the whole step as a box.
    ///
    /// The dynamic part needs boxes of `s`, `δ` and `Δz`. The step
    /// takes `s` from the interval hull of `X − x_m`, `Δz` from the hull of the
    /// time-interval set `R(τ)` and `U`, the input part of `δ` from `U − u*`, and
    /// the state part from the flow:
    /// `x(t) − x_k = ∫_{t_k}^t f(z(σ)) dσ ∈ [0, r] · f(□R(τ) × □U)`, minus `c`
    /// (enclosed in interval arithmetic), as long as the trajectory stays in
    /// `R(τ)` up to `t`. That is the same hypothesis the
    /// guess-and-check loop proves for every other term, so the usual argument
    /// closes: if the dynamic error computed over `R(τ)` lies in the guess `Ψ̄`,
    /// no trajectory can leave `R(τ)` during the step, because up to the first
    /// exit its error lies in `Ψ̄` and the linear model with an input in `Ψ̄`
    /// keeps it in `R(τ)`. The remainder evaluates `∂³f` over a box that holds
    /// every segment from `z*` into `z* + □Δz`, which holds every `ξ(t)`.
    ///
    /// The decomposition is exact algebra, every box encloses its quantity, and
    /// every product and sum of boxes is interval arithmetic, outward rounded
    /// with the `sound` feature. Removing any one term makes the claim false;
    /// the unit tests show that for each term that floating-point rounding does
    /// not dominate.
    Polynomialization,
}

/// How [`NonlinearReach::reach`] splits a piece whose step diverged.
///
/// Non-exhaustive: more strategies may be added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Splitting {
    /// Factor splits and box splits compete, and the one whose larger half
    /// has the smaller second-order error bound wins, split by split. The
    /// default: on the cubic drift `x₀' = x₀³` from radius 0.5, factor
    /// splits alone needed 30 pieces where this needs 4.
    #[default]
    Cheapest,
    /// Factor splits only: every piece keeps all dependencies of the start
    /// set, at the price of more pieces.
    Factors,
}

/// What becomes of the generators that order reduction and the linear step
/// make independent.
///
/// Non-exhaustive: more restructurings may be added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Restructure {
    /// They stay independent generators. Their products in the next step's
    /// quadratic error are enclosed one by one, so a wide set loses its
    /// shape step by step.
    Off,
    /// After each step's Girard reduction, every independent generator of
    /// the time-point set becomes a dependent factor of its own. That is
    /// exact (`g β` with `β ∈ [-1, 1]` is the monomial `g α` of a factor
    /// nowhere else), and from then on the quadratic maps of later steps
    /// treat it as a polynomial: `α²` stays non-negative and products
    /// cancel. The default: on Van der Pol from radius 1 it proves 20
    /// steps with polynomialization instead of 15, and 13 with
    /// linearization instead of 8, at 2 to 3 times the cost per step. The
    /// number of factors stays at most the number of generators, since
    /// [`compact`](polyzonotope::SparsePolyZonotope::compact) drops the
    /// unused ones.
    #[default]
    Lift,
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
    /// Which splits [`reach`](NonlinearReach::reach) may use.
    pub splitting: Splitting,
    /// The abstraction of each step.
    pub method: Method,
    /// How much larger than the two pieces' interval hulls together the
    /// hull of their merged enclosure may be, by volume, as a fraction (0.1
    /// allows 10%), for [`reach`](NonlinearReach::reach) to merge them into
    /// one piece for the next step. Non-negative. A merge is also never
    /// wider than half the narrowest piece that diverged so far, so splits
    /// are not undone while the pieces are still too large to step whole.
    pub merge_tolerance: f64,
    /// Highest total degree the static error of
    /// [`Method::Polynomialization`] keeps exact, at least 1. It is
    /// quadratic in the set, so without a cap the degree of the sets would
    /// double with every step; higher monomials are enclosed in independent
    /// generators.
    pub max_degree: u32,
    /// Whether the time-point set's independent generators become
    /// dependent factors before the next step.
    pub restructure: Restructure,
    /// Adaptive step lengths for [`reach`](NonlinearReach::reach) and
    /// [`reach_for`](NonlinearReach::reach_for), starting from `step`; `None`
    /// keeps every step at `step`.
    pub adaptive: Option<Adaptive>,
}

impl NonlinearSettings {
    /// Defaults for time step `step`: 4 Taylor terms, order 20, 10
    /// iterations, 10% widening, up to 8 pieces from the cheapest splits,
    /// merged within 10%, lifted generators, fixed steps,
    /// linearization, and degree 4 for polynomialization.
    #[must_use]
    pub fn new(step: f64) -> Self {
        Self {
            step,
            taylor_terms: 4,
            order: 20.0,
            max_iterations: 10,
            enlarge: 0.1,
            max_pieces: 8,
            splitting: Splitting::Cheapest,
            method: Method::Linearization,
            merge_tolerance: 0.1,
            max_degree: 4,
            restructure: Restructure::Lift,
            adaptive: None,
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
    /// `enlarge` or `merge_tolerance` is negative, `max_pieces` or
    /// `max_degree` is zero, or, with [`Adaptive`] steps, `min_step` is not
    /// positive, `max_step` not finite, `step` not in
    /// `[min_step, max_step]`, `shrink` not in `(0, 1)` or `grow` below 1.
    #[must_use]
    pub fn new(dynamics: D, settings: NonlinearSettings) -> Self {
        assert!(
            settings.step > 0.0 && settings.step.is_finite(),
            "step must be positive"
        );
        assert!(settings.order >= 1.0, "order must be at least 1");
        assert!(settings.enlarge >= 0.0, "enlarge must not be negative");
        assert!(
            settings.merge_tolerance >= 0.0,
            "merge_tolerance must not be negative"
        );
        assert!(settings.max_pieces >= 1, "max_pieces must be at least 1");
        assert!(settings.max_degree >= 1, "max_degree must be at least 1");
        if let Some(adaptive) = &settings.adaptive {
            adaptive.check(settings.step);
        }

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
        let attempt = Attempt {
            step: self.settings.step,
            terms: Terms::ALL,
        };
        self.step_with(x, u, attempt).map(|(step, _)| step)
    }

    /// [`step`](Self::step) with the step length and error terms of
    /// `attempt`, and how many error guesses it took; the tests drop one term
    /// at a time to show that each is needed.
    fn step_with(
        &self,
        x: &SparsePolyZonotope,
        u: &Zonotope,
        attempt: Attempt,
    ) -> Result<(Step, usize), ReachError> {
        assert_eq!(x.dim(), self.dynamics.states(), "x needs one row per state");
        assert_eq!(u.dim(), self.dynamics.inputs(), "u needs one row per input");

        let model = self
            .linearize(x, u, attempt.step)
            .ok_or(ReachError::AbstractionDiverged)?;
        match self.settings.method {
            Method::Linearization => self.step_linearized(x, u, &model, attempt),
            Method::Polynomialization => self.step_polynomialized(x, u, &model, attempt),
        }
    }

    /// One step of [`Method::Linearization`]: the linear model with the
    /// whole error as a guessed box.
    fn step_linearized(
        &self,
        x: &SparsePolyZonotope,
        u: &Zonotope,
        model: &Linearization,
        attempt: Attempt,
    ) -> Result<(Step, usize), ReachError> {
        let n = self.dynamics.states();
        let system = LinearSystem {
            a: model.a.clone(),
            b: DMatrix::identity(n, n),
        };
        let linear = LinearReach::new(system, attempt.step, self.settings.taylor_terms)?;
        let known = u.linear_map(&model.b);

        self.confirm(n, attempt.step, |assumed| {
            let step = linear.step(x, &with_error(&known, &model.offset, assumed));
            let states = step
                .time_interval
                .reduce(self.settings.order, Reduction::Girard);
            let actual =
                error::abstraction_error(&self.dynamics, model, (&states, u), attempt.terms)
                    .ok_or(ReachError::AbstractionDiverged)?;

            Ok((step.time_point, states, actual))
        })
    }

    /// One step of [`Method::Polynomialization`]: the static error `Ψ_s`
    /// rides along as the constant input of an augmented linear system, and
    /// only the dynamic error is a guessed box. The soundness argument is on
    /// [`Method::Polynomialization`].
    fn step_polynomialized(
        &self,
        x: &SparsePolyZonotope,
        u: &Zonotope,
        model: &Linearization,
        attempt: Attempt,
    ) -> Result<(Step, usize), ReachError> {
        let n = self.dynamics.states();
        let limits = (self.settings.max_degree, self.settings.order);
        let expansion = polynomialization::static_error(&self.dynamics, model, x, limits)?;
        let start = if attempt.terms.has(Term::StaticSet) {
            x.cartesian_product(&expansion.static_set)
        } else {
            x.cartesian_product(&SparsePolyZonotope::from_zonotope(&Zonotope::new(
                DVector::zeros(n),
                DMatrix::zeros(n, 0),
            )))
        };
        let system = polynomialization::augmented_system(&model.a);
        let linear = LinearReach::new(system, attempt.step, self.settings.taylor_terms)?;
        let known = u.linear_map(&model.b);

        self.confirm(n, attempt.step, |assumed| {
            let step = linear.step(&start, &with_error(&known, &model.offset, assumed));
            let states = polynomialization::project(&step.time_interval, n)
                .reduce(self.settings.order, Reduction::Girard);
            let actual = polynomialization::dynamic_error(
                &self.dynamics,
                (model, &expansion),
                (&states, u),
                attempt.step,
                attempt.terms,
            )?;

            Ok((
                polynomialization::project(&step.time_point, n),
                states,
                actual,
            ))
        })
    }

    /// `set` reduced to the configured order, as the start of the next step,
    /// with its independent generators lifted to factors under
    /// [`Restructure::Lift`].
    fn restructure(&self, set: &SparsePolyZonotope) -> SparsePolyZonotope {
        let reduced = set.reduce(self.settings.order, Reduction::Girard);
        match self.settings.restructure {
            Restructure::Off => reduced,
            Restructure::Lift => split::lift_independent(&reduced),
        }
    }

    /// The guess-and-check loop: `attempt` runs the step with the error
    /// guess `Ψ̄` and returns the time-point set, the reduced time-interval
    /// set and the error over it. The first attempt whose error lies in its
    /// guess is accepted, with the number of guesses it took; otherwise the
    /// guess widens, up to `max_iterations` times. `step` is the length of
    /// the step, recorded in the result.
    ///
    /// # Errors
    ///
    /// The first error of `attempt`, or [`ReachError::AbstractionDiverged`]
    /// if no guess was confirmed or the guess stopped being finite.
    fn confirm(
        &self,
        n: usize,
        step: f64,
        mut attempt: impl FnMut(
            &[Interval],
        ) -> Result<
            (SparsePolyZonotope, SparsePolyZonotope, Vec<Interval>),
            ReachError,
        >,
    ) -> Result<(Step, usize), ReachError> {
        let mut assumed = vec![Interval::point(0.0); n];
        for guesses in 1..=self.settings.max_iterations {
            let (time_point, time_interval, actual) = attempt(&assumed)?;
            if encloses(&assumed, &actual) {
                let step = Step {
                    time_point: self.restructure(&time_point),
                    time_interval,
                    duration: step,
                };
                return Ok((step, guesses));
            }

            assumed = widen(&assumed, &actual, self.settings.enlarge);
            if !all_finite(assumed.iter()) {
                break;
            }
        }

        Err(ReachError::AbstractionDiverged)
    }

    /// The linear model around `z* = (x* , c_u)`, where `x*` is the middle
    /// of `x`'s interval hull moved half a step along `f` (as CORA does);
    /// `None` if a value along the way is not finite.
    fn linearize(&self, x: &SparsePolyZonotope, u: &Zonotope, step: f64) -> Option<Linearization> {
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
            .map(|(&m, d)| m + 0.5 * step * f64::midpoint(d.lo, d.hi))
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

/// The step length and error terms of one step attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Attempt {
    /// The step length `r`, positive and finite.
    pub(crate) step: f64,
    /// The error terms to include: all of them outside the tests.
    pub(crate) terms: Terms,
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

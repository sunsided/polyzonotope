//! The multi-step driver: steps every piece, splits what diverges, merges
//! what has become small, and with [`Adaptive`] steps adapts the step
//! length.

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{Interval, SparsePolyZonotope, Zonotope};

use super::{Attempt, NonlinearReach, Terms, polynomialization, split};
use crate::{Dynamics, ReachError, ReachFailure, Step};

/// Adaptive step lengths: shrink the step when a time step is refused, and
/// grow it after easy ones.
///
/// [`reach`](NonlinearReach::reach) and
/// [`reach_for`](NonlinearReach::reach_for) start with
/// [`NonlinearSettings::step`](super::NonlinearSettings::step). When a piece
/// diverges or the Taylor bound fails, the whole time step is retried with
/// the step times `shrink`, but never shorter than `min_step`; only at
/// `min_step` itself are pieces split. After a time step whose pieces all
/// confirmed their error within `easy_guesses` guesses, the next step is
/// `grow` times longer, at most `max_step`. Each [`Step`] records its
/// [`duration`](Step::duration).
///
/// # Soundness
///
/// Nothing in a single step depends on the step length being the same as
/// before: the Taylor series, its remainder, the time-interval corrections
/// and the error check are all computed for the `r` of that attempt, and
/// the dynamics do not depend on time. A refused attempt leaves no set
/// behind, so the retry starts from the same proven sets. Step `k` covers
/// `[t_k, t_k + r_k]` with `t_{k+1} = t_k + r_k`, starting from the
/// time-point sets of step `k − 1`, so the time-interval sets together cover
/// every time from 0 to the sum of the durations, and the time-point set of
/// the last step holds every state at that time. The durations are the
/// exact `f64` values the steps used; only their sum, if a caller adds them
/// up, rounds.
///
/// # Examples
///
/// ```
/// use nalgebra::{DMatrix, DVector};
/// use polyzonotope_reach::polyzonotope::{SparsePolyZonotope, Zonotope};
/// use polyzonotope_reach::{Adaptive, NonlinearReach, NonlinearSettings, Polynomial};
///
/// let vdp = Polynomial::autonomous(2)
///     .term(0, 1.0, &[0, 1])
///     .term(1, 1.0, &[0, 1])
///     .term(1, -1.0, &[2, 1])
///     .term(1, -1.0, &[1, 0]);
/// let settings = NonlinearSettings {
///     adaptive: Some(Adaptive::new(0.001, 0.1)),
///     ..NonlinearSettings::new(0.02)
/// };
/// let start = SparsePolyZonotope::from_zonotope(&Zonotope::new(
///     DVector::from_vec(vec![1.4, 2.4]),
///     DMatrix::from_diagonal_element(2, 2, 0.05),
/// ));
/// let no_input = Zonotope::new(DVector::zeros(0), DMatrix::zeros(0, 0));
///
/// let flowpipe = NonlinearReach::new(vdp, settings).reach_for(&start, &no_input, 0.5)?;
/// let total: f64 = flowpipe.iter().map(|pieces| pieces[0].duration).sum();
/// assert!((total - 0.5).abs() < 1e-12);
/// # Ok::<(), polyzonotope_reach::ReachFailure>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Adaptive {
    /// Shortest step, positive; at this length a refusal splits the
    /// pieces instead of shrinking further.
    pub min_step: f64,
    /// Longest step, at least `min_step`.
    pub max_step: f64,
    /// Factor a refused step shrinks by, in `(0, 1)`.
    pub shrink: f64,
    /// Factor an easy step grows the next one by, at least 1.
    pub grow: f64,
    /// Most error guesses a piece may need for its step to count as easy.
    pub easy_guesses: usize,
}

impl Adaptive {
    /// Steps between `min_step` and `max_step` that halve on a refusal and
    /// grow by a quarter after a step whose errors were confirmed within
    /// three guesses.
    #[must_use]
    pub fn new(min_step: f64, max_step: f64) -> Self {
        Self {
            min_step,
            max_step,
            shrink: 0.5,
            grow: 1.25,
            easy_guesses: 3,
        }
    }

    /// Asserts that the settings make sense for an initial step `step`.
    ///
    /// # Panics
    ///
    /// Panics if `min_step` is not positive and finite, `step` is not within
    /// `[min_step, max_step]`, `shrink` is not in `(0, 1)`, or `grow` is
    /// below 1.
    pub(super) fn check(&self, step: f64) {
        assert!(
            self.min_step > 0.0 && self.max_step.is_finite(),
            "adaptive steps must be positive and finite"
        );
        assert!(
            self.min_step <= step && step <= self.max_step,
            "the initial step must lie in [min_step, max_step]"
        );
        assert!(
            self.shrink > 0.0 && self.shrink < 1.0,
            "shrink must lie in (0, 1)"
        );
        assert!(self.grow >= 1.0, "grow must be at least 1");
    }
}

/// When a run stops.
#[derive(Clone, Copy, Debug)]
enum Horizon {
    /// After this many steps.
    Steps(usize),
    /// Once the steps add up to this duration; the last step is cut short.
    Time(f64),
}

/// The state of a run between time steps.
struct Run {
    /// The proven steps so far.
    flowpipe: Vec<Vec<Step>>,
    /// The start sets of the next step.
    pieces: Vec<SparsePolyZonotope>,
    /// The width of the narrowest piece that diverged so far.
    diverged: f64,
    /// The step length to try next.
    step: f64,
    /// The sum of the durations so far.
    elapsed: f64,
}

/// One proven time step, from [`NonlinearReach::advance`].
struct Advanced {
    /// The steps of all pieces.
    steps: Vec<Step>,
    /// The width of the narrowest piece that diverged and was split,
    /// infinite if none.
    narrowest: f64,
    /// The most error guesses any piece needed.
    guesses: usize,
}

impl<D: Dynamics> NonlinearReach<D> {
    /// `steps` consecutive steps from `x0`: entry `k` holds the pieces of
    /// step `k`, whose union encloses the reachable set over
    /// `[t_k, t_k + r_k]`, with `r_k` the step's
    /// [`duration`](Step::duration): the configured step, unless
    /// [`adaptive`](super::NonlinearSettings::adaptive) varies it.
    ///
    /// A piece whose step diverges is split in two and retried, as long as
    /// the step stays within `max_pieces` (with adaptive steps, only once the
    /// step is at its shortest). The split cuts one factor of the piece,
    /// keeping its dependencies, or (unless
    /// [`Splitting::Factors`](super::Splitting::Factors)) one axis of its
    /// oriented box, whichever leaves halves with the smaller second-order
    /// error bound. After each step, pieces whose merged enclosure is within
    /// `merge_tolerance` of their hulls, and no wider than half the narrowest
    /// piece that diverged so far, become one piece again.
    ///
    /// # Errors
    ///
    /// A [`ReachFailure`] with the first error of [`step`](Self::step) that
    /// splitting and shrinking could not avoid, and every step proven
    /// before it.
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
        self.run(x0, u, Horizon::Steps(steps))
    }

    /// Steps from `x0` until their durations add up to `duration`, the last
    /// one cut short to end there; otherwise as [`reach`](Self::reach).
    /// Mostly useful with [`adaptive`](super::NonlinearSettings::adaptive)
    /// steps, where the number of steps is not known in advance.
    ///
    /// # Errors
    ///
    /// As for [`reach`](Self::reach).
    ///
    /// # Panics
    ///
    /// Panics if `duration` is negative or not finite, or if `x0` does not
    /// have one row per state or `u` one per input.
    pub fn reach_for(
        &self,
        x0: &SparsePolyZonotope,
        u: &Zonotope,
        duration: f64,
    ) -> Result<Vec<Vec<Step>>, ReachFailure> {
        assert!(
            duration >= 0.0 && duration.is_finite(),
            "duration must be non-negative and finite"
        );
        self.run(x0, u, Horizon::Time(duration))
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

    /// Time steps from `x0` until `horizon`.
    ///
    /// # Errors
    ///
    /// A [`ReachFailure`] with the steps proven before the one that failed.
    fn run(
        &self,
        x0: &SparsePolyZonotope,
        u: &Zonotope,
        horizon: Horizon,
    ) -> Result<Vec<Vec<Step>>, ReachFailure> {
        let mut run = Run {
            flowpipe: Vec::new(),
            pieces: vec![x0.clone()],
            diverged: f64::INFINITY,
            step: self.settings.step,
            elapsed: 0.0,
        };
        while let Some(length) = next_length(horizon, &run) {
            if let Err(error) = self.time_step(&mut run, u, length) {
                return Err(ReachFailure {
                    proven: run.flowpipe,
                    error,
                });
            }
        }

        Ok(run.flowpipe)
    }

    /// One time step of `run`, at most `length` long: shrinks and retries
    /// on a refusal while the adaptive settings allow, then records the
    /// step, adapts the next length and prepares the next pieces.
    ///
    /// # Errors
    ///
    /// The error of the last attempt if neither shrinking nor splitting
    /// helped.
    fn time_step(&self, run: &mut Run, u: &Zonotope, mut length: f64) -> Result<(), ReachError> {
        let adaptive = self.settings.adaptive.as_ref();
        loop {
            let shrinkable = adaptive.is_some_and(|a| length > a.min_step);
            match self.advance(run.pieces.clone(), u, length, !shrinkable) {
                Ok(advanced) => {
                    run.step = grown(adaptive, run.step.min(length), advanced.guesses);
                    run.elapsed += length;
                    run.diverged = run.diverged.min(advanced.narrowest);
                    run.pieces = self.next_pieces(&advanced.steps, 0.5 * run.diverged);
                    run.flowpipe.push(advanced.steps);
                    return Ok(());
                }
                Err(ReachError::AbstractionDiverged | ReachError::StepTooLarge) if shrinkable => {
                    // Clamped, so that `min_step` itself is tried before pieces split.
                    length = adaptive.map_or(length, |a| (length * a.shrink).max(a.min_step));
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// One step of length `length` for every piece in `todo`, splitting
    /// pieces that diverge if `may_split`.
    ///
    /// # Errors
    ///
    /// The first error of a piece that splitting could not avoid.
    fn advance(
        &self,
        mut todo: Vec<SparsePolyZonotope>,
        u: &Zonotope,
        length: f64,
        may_split: bool,
    ) -> Result<Advanced, ReachError> {
        let attempt = Attempt {
            step: length,
            terms: Terms::ALL,
        };
        let mut done = Advanced {
            steps: Vec::with_capacity(todo.len()),
            narrowest: f64::INFINITY,
            guesses: 0,
        };
        while let Some(piece) = todo.pop() {
            match self.step_with(&piece, u, attempt) {
                Ok((step, guesses)) => {
                    done.guesses = done.guesses.max(guesses);
                    done.steps.push(step);
                }
                Err(ReachError::AbstractionDiverged)
                    if may_split
                        && done.steps.len() + todo.len() + 2 <= self.settings.max_pieces =>
                {
                    done.narrowest = done.narrowest.min(split::width(&piece));
                    let (lower, upper) =
                        split::halves(&piece, |p| self.curvature(p, u), self.settings.splitting)
                            .ok_or(ReachError::AbstractionDiverged)?;
                    todo.push(lower);
                    todo.push(upper);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(done)
    }

    /// `Σ_i |r|ᵀ |H_i| |r|`, with `r` the radius of the hull of `x × u` and
    /// `H_i` the Hessians over it: a bound on the second-order error of a
    /// linearization of `x`, the size that splitting has to bring down.
    /// Plain `f64`, since it only ranks candidate splits; infinite if the
    /// Hessians overflow.
    fn curvature(&self, x: &SparsePolyZonotope, u: &Zonotope) -> f64 {
        let region: Vec<Interval> = x
            .interval_hull()
            .0
            .into_iter()
            .chain(u.interval_hull().0)
            .collect();
        let radius: Vec<f64> = region.iter().map(|iv| 0.5 * iv.width()).collect();
        let w = radius.len();
        let bound: f64 = self
            .dynamics
            .hessians(&region)
            .iter()
            .flat_map(|h| {
                let radius = &radius;
                (0..w).flat_map(move |a| {
                    (0..w).map(move |b| h.0[(a, b)].mag() * radius[a] * radius[b])
                })
            })
            .sum();

        if bound.is_nan() { f64::INFINITY } else { bound }
    }

    /// The start sets of the next step: the time-point sets of `steps`,
    /// with pairs merged where [`merge_tolerance`](super::NonlinearSettings::merge_tolerance)
    /// allows and the merge is no wider than `cap`.
    fn next_pieces(&self, steps: &[Step], cap: f64) -> Vec<SparsePolyZonotope> {
        let pieces = steps.iter().map(|s| s.time_point.clone()).collect();
        split::merge(pieces, self.settings.merge_tolerance, cap, |merged| {
            self.restructure(&polynomialization::cap_degree(
                merged,
                self.settings.max_degree,
            ))
        })
    }
}

/// The length of the next time step of `run`, or `None` once `horizon` is
/// reached: the run's step, cut to the time left for a duration. A step
/// that leaves only a rounding residue is stretched to the end instead, so
/// that ten steps of 0.1, which add up to 0.9999999999999999, do not need
/// an eleventh step of 1e-16.
fn next_length(horizon: Horizon, run: &Run) -> Option<f64> {
    match horizon {
        Horizon::Steps(n) => (run.flowpipe.len() < n).then_some(run.step),
        Horizon::Time(t) => {
            let left = t - run.elapsed;
            let length = if left <= run.step * (1.0 + 1e-9) {
                left
            } else {
                run.step
            };
            (left > 0.0).then_some(length)
        }
    }
}

/// The step after one of length `step` whose pieces needed at most
/// `guesses` error guesses: grown if that was easy, else unchanged.
fn grown(adaptive: Option<&Adaptive>, step: f64, guesses: usize) -> f64 {
    match adaptive {
        Some(a) if guesses <= a.easy_guesses => (step * a.grow).min(a.max_step),
        _ => step,
    }
}

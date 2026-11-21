#![doc = include_str!("../README.md")]
#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

/// The set types this crate propagates, re-exported so callers need one dependency.
pub use polyzonotope;

mod dynamics;
mod linear;
mod nonlinear;
mod taylor;

pub use dynamics::{Dynamics, IntervalTensor, Polynomial};
pub use linear::{LinearReach, LinearSystem, Step};
pub use nonlinear::{Adaptive, Method, NonlinearReach, NonlinearSettings, Restructure, Splitting};

/// Why a reachability computation could not produce a sound result.
///
/// Non-exhaustive: new ways to refuse may be added without a breaking
/// release, so match with a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReachError {
    /// The time step is negative, infinite or NaN; the remainder bound needs
    /// a finite `r >= 0`.
    InvalidStep,
    /// The time step is too large for the number of Taylor terms: the
    /// remainder bound of the matrix exponential needs `‖A‖∞ r < η + 2`.
    StepTooLarge,
    /// The abstraction error of a nonlinear step was not confirmed: it kept
    /// outgrowing the assumed bound, or stopped being finite. The set or
    /// the step is too large for the abstraction (linearization or
    /// polynomialization); split the set or use a smaller step.
    AbstractionDiverged,
    /// [`Method::Polynomialization`] needs third derivatives for its
    /// Lagrange remainder, and [`Dynamics::third_derivatives`] supplies none.
    /// Implement it, or use [`Method::Linearization`].
    MissingDerivatives,
}

impl core::fmt::Display for ReachError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidStep => "time step must be finite and non-negative",
            Self::StepTooLarge => "time step too large for the Taylor remainder bound",
            Self::AbstractionDiverged => "abstraction error of the step did not converge",
            Self::MissingDerivatives => "polynomialization needs third derivatives of the dynamics",
        })
    }
}

impl core::error::Error for ReachError {}

/// A multi-step computation that stopped early, with everything it proved
/// before the failing step.
///
/// Returned by [`NonlinearReach::reach`]: the sets in `proven` are as sound
/// as those of a complete run, they just end before the requested horizon.
#[derive(Clone, Debug, PartialEq)]
pub struct ReachFailure {
    /// The completed steps, in order: entry `k` holds the pieces of step `k`,
    /// exactly as a successful run would have returned them. A step that
    /// failed for any of its pieces is left out entirely, so this is shorter
    /// than the requested horizon and may be empty.
    pub proven: alloc::vec::Vec<alloc::vec::Vec<Step>>,
    /// Why the next step could not be proven.
    pub error: ReachError,
}

impl core::fmt::Display for ReachFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} after {} proven steps", self.error, self.proven.len())
    }
}

impl core::error::Error for ReachFailure {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}

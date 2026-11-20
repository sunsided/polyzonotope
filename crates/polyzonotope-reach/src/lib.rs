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

pub use dynamics::{Dynamics, Polynomial};
pub use linear::{LinearReach, LinearSystem, Step};
pub use nonlinear::{NonlinearReach, NonlinearSettings};

/// Why a reachability computation could not produce a sound result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReachError {
    /// The time step is negative, infinite or NaN; the remainder bound needs
    /// a finite `r >= 0`.
    InvalidStep,
    /// The time step is too large for the number of Taylor terms: the
    /// remainder bound of the matrix exponential needs `‖A‖∞ r < η + 2`.
    StepTooLarge,
    /// The abstraction error of a nonlinear step was not confirmed: it kept
    /// outgrowing the assumed bound, or stopped being finite. The set or
    /// the step is too large for the linearization; split the set or use a
    /// smaller step.
    AbstractionDiverged,
}

impl core::fmt::Display for ReachError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidStep => "time step must be finite and non-negative",
            Self::StepTooLarge => "time step too large for the Taylor remainder bound",
            Self::AbstractionDiverged => "abstraction error of the linearization did not converge",
        })
    }
}

impl core::error::Error for ReachError {}

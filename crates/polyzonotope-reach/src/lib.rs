#![doc = include_str!("../README.md")]
#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

/// The set types this crate propagates, re-exported so callers need one dependency.
pub use polyzonotope;

mod linear;
mod taylor;

pub use linear::{LinearReach, LinearSystem, Step};

/// Why a reachability computation could not produce a sound result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReachError {
    /// The time step is too large for the number of Taylor terms: the
    /// remainder bound of the matrix exponential needs `‖A‖∞ r < η + 2`.
    StepTooLarge,
}

impl core::fmt::Display for ReachError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::StepTooLarge => "time step too large for the Taylor remainder bound",
        })
    }
}

impl core::error::Error for ReachError {}

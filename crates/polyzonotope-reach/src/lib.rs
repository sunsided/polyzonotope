#![doc = include_str!("../README.md")]
#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

/// The set types this crate propagates, re-exported so callers need one dependency.
pub use polyzonotope;

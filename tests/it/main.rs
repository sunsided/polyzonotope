//! Integration tests: soundness and exactness properties on random sets, plus edge cases.

mod edge_cases;
mod properties;
#[cfg(feature = "sound")]
mod sound;
mod strategies;

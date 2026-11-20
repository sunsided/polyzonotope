//! Integration tests: soundness and exactness properties on random sets, plus edge cases.

mod constrained;
mod cora;
mod edge_cases;
mod properties;
#[cfg(feature = "sound")]
mod sound;
mod strategies;

---
title: Rust code
applyTo: "**/*.rs"
summary: House style: docs on every item, no mod.rs, no separators, blank lines, helper extraction, pedantic clippy, no unsafe.
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
- Soundness first: an operation documented as an enclosure (`interval_hull`, `to_zonotope`)
  must contain every point of the input set. An operation documented as exact must commute
  with evaluation. Flag any change that weakens either claim.
- Every new set operation gets a property in `tests/it/properties.rs` that checks its
  exactness or enclosure claim on random sets from `tests/it/strategies.rs`.
- Dependent factors are identified by `FactorId`, not by position. Operations that combine
  sets (`exact_plus`, `cartesian_product`) keep shared ids correlated. Only
  `minkowski_sum` and `with_fresh_ids` decorrelate on purpose.
- Arithmetic is plain `f64` without outward rounding (see the README's soundness caveat). Do
  not claim formal soundness in docs until the interval mode lands.

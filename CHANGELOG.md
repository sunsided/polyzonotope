# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/sunsided/polyzonotope/compare/v0.1.0...v0.2.0) - 2026-11-20

### Added

- Add constrained polynomial zonotopes
- Add plotters examples and an RRT that keeps only proven-safe edges
- Add three-valued intersection and containment checks
- Add interval_hull_split, a hull tightened by branch and bound
- Add Girard and PCA order reduction
- Add quadratic_map for sparse polynomial zonotopes
- Add a sound mode with outward rounding via inari
- Support no_std with alloc
- Add the semver check to CI now that 0.1.0 is published

### Changed

- Word the unsplittable case of the checks accurately
- Word the no-factor case of the split hull accurately
- Address review: guard the CORA case count, match extractAfter
- Address review: plain-mode feasibility slack, outward slack halving
- Address review: unsplittable factors, README table, Unknown wording
- Skip factors split cannot take in the split hull
- Keep the halfspace constraint on the set's own β
- Sort polygon generators by the sign of the cross product
- Say that the proof verdicts are rigorous only with sound
- Cross-check enclosures and reductions against CORA
- Fall back to the plain hull when rounding crosses the split bounds
- Check Girard's box against the exact radius in sound mode
- Check quadratic_map against the exact square in sound mode
- Address review: exponent limit, compact growth, sound docs
- Check sound results against the stored error radius exactly
- Cover shape errors and the zonotope operations in tests
- Test the power helpers directly
- Check the no_std build in CI and document it
- Assert the β count in zonotope_coefficients
- Relate reachability to RRT planning in the README
- Index zonotope coefficients cyclically in from_zonotope_is_exact
- Leave zero generators out of to_zonotope
- Test from_zonotope, Zonotope::order and the new panics
- Explain in the README what the crate is for
- Run the semver check in task ci
- Reject NaN points and the order of zero-dimensional zonotopes
- Evaluate positional factor values without a HashMap
- Make the Zonotope fields private
- Test the Minkowski sum against independent factor values
- Keep independent generators in compact

### Dependencies

- Bump the version to 0.2.0 for the private Zonotope fields

### Removed

- Drop needless clones and repeated column builders

## [0.1.0](https://github.com/sunsided/polyzonotope/releases/tag/v0.1.0) - 2026-11-19

### Added

- Add strict lint table, clippy and rustfmt configuration
- Add release-plz and tag-triggered crates.io publishing
- Add CONTRIBUTING.md
- Add prek hooks and Copilot review instructions
- Add crate keywords, categories and README badges
- Add REUSE licensing metadata and a README license section
- Add cargo-deny, typos and Dependabot configuration
- Add Codecov configuration
- Add Taskfile and split CI into per-check jobs
- Add README
- Add a quadratic dependency example
- Add property tests
- Add sparse polynomial zonotopes
- Add zonotopes
- Add intervals and interval boxes

### Changed

- Report mermaid blocks without a closing fence
- Refuse to publish tags that are not on main
- Document every branch prefix the branch check accepts
- Run actionlint from its official image in CI
- Give release PRs an imperative title
- Separate statement blocks with blank lines in sparse.rs
- Document every item and fix pedantic clippy findings
- Use the README as crate docs and render mermaid on docs.rs
- Move property tests to tests/it and add criterion benches
- Update nalgebra to 0.35 and raise MSRV to 1.89
- Initial commit

### Fixed

- Fix β alignment in the split property
- Fix stale and dangling comments in workflows and configs

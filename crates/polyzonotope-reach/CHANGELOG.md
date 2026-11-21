# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/sunsided/polyzonotope/releases/tag/polyzonotope-reach-v0.1.0) - 2026-11-21

### Added

- Add factor-only splitting and show the soundness argument in the docs
- Add conservative polynomialization to the nonlinear reach
- Add a benchmark for nonlinear reachability
- Add nonlinear reachability by conservative linearization
- Add a Dynamics trait and polynomial dynamics to the reach crate
- Add linear reachability to polyzonotope-reach

### Changed

- Reject max_iterations of zero
- Hold the merge cap for the finished piece
- Try min_step itself before splitting pieces
- Refuse an overflowing linearization offset instead of panicking
- Test the damped cubic with lifting and a larger piece budget
- Record the failed reduction and restructuring trials in the roadmap
- Lift reduced generators to factors and add adaptive time steps
- Document polynomialization, splitting and merging in the README
- Split by factors or boxes, whichever is cheaper, and merge pieces
- Return proven steps on failure and add autonomous helpers
- Refuse instead of panicking when the dynamics overflow
- Check both sides of each direction and bound ‖A‖∞ r in the tests
- Refuse negative and non-finite time steps
- Round the constant input of linear reach outward
- Turn the repository into a workspace with polyzonotope-reach

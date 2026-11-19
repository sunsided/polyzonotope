# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

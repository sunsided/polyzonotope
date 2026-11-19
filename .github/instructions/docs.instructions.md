---
title: Documentation
applyTo: "**/*.md,docs/**"
summary: Absolute URLs, valid and current mermaid, doctested code blocks.
---

# Documentation

- README files are rendered on crates.io and docs.rs. Every link to a repo file and every
  image uses an absolute URL (`https://github.com/...`, `https://raw.githubusercontent.com/...`).
  Only in-page anchors may be relative.
- Mermaid diagrams (```` ```mermaid ````) must be valid (`task lint:mermaid` renders them)
  and must match the code. Flag diagrams that the change makes stale. Keep each diagram under
  about 15 nodes.
- Code blocks are tagged with their language. Untagged or `rust` blocks in included Markdown
  run as doctests, so they must compile. Use `rust,no_run` or `rust,ignore` only with a
  stated reason.
- `README.md` is the crate-level documentation (`#![doc = include_str!("../README.md")]` in
  `src/lib.rs`). Its Rust blocks are doctests, and it renders on docs.rs. Write it for users
  of the crate.
- The operations table in the README (exact vs. sound enclosure) and the conversion diagram
  must match the API. A new operation gets a row.
- Keep badges, the License section and the MSRV statement consistent with `Cargo.toml`.

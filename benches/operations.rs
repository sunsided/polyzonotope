//! Benchmarks for the hot set operations on a mid-sized sparse polynomial zonotope.

use std::hint::black_box;

use criterion::Criterion;
use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, SparsePolyZonotope};

/// Dimension of the benchmark set.
const DIM: usize = 4;

/// Number of dependent generators of the benchmark set.
const DEPENDENT: usize = 24;

/// Number of independent generators of the benchmark set.
const INDEPENDENT: usize = 4;

/// Number of dependent factors of the benchmark set.
const FACTORS: usize = 5;

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    let pz = benchmark_set();

    bench_enclosures(&mut criterion, &pz);
    bench_exact_operations(&mut criterion, &pz);
    criterion.final_summary();
}

/// A deterministic set with mixed even and odd exponents in `0..=3`, so every
/// monomial range of `to_zonotope` and `interval_hull` is exercised.
fn benchmark_set() -> SparsePolyZonotope {
    let center = DVector::from_fn(DIM, |i, _| pseudo_random(i, 0));
    let dependent = DMatrix::from_fn(DIM, DEPENDENT, |i, j| pseudo_random(i, j + 1));
    let independent = DMatrix::from_fn(DIM, INDEPENDENT, |i, j| pseudo_random(i, j + 97));
    let exponents = DMatrix::from_fn(FACTORS, DEPENDENT, |k, j| {
        u32::try_from((k * 7 + j * 3) % 4).expect("value is below 4")
    });

    SparsePolyZonotope::new(
        center,
        dependent,
        independent,
        exponents,
        FactorId::fresh_n(FACTORS),
    )
    .expect("shapes are consistent by construction")
}

/// A fixed value in `[-1, 1)` for the matrix entry `(row, col)`.
fn pseudo_random(row: usize, col: usize) -> f64 {
    let hash = (row * 31 + col * 17) % 101;
    let hash = u32::try_from(hash).expect("value is below 101");

    f64::from(hash) / 50.5 - 1.0
}

/// Benchmarks the sound enclosures: interval hull and zonotope.
fn bench_enclosures(criterion: &mut Criterion, pz: &SparsePolyZonotope) {
    criterion.bench_function("interval_hull", |b| {
        b.iter(|| black_box(pz).interval_hull());
    });
    criterion.bench_function("to_zonotope", |b| b.iter(|| black_box(pz).to_zonotope()));
}

/// Benchmarks the exact operations that grow or reshape the representation.
fn bench_exact_operations(criterion: &mut Criterion, pz: &SparsePolyZonotope) {
    let other = pz.linear_map(&DMatrix::from_fn(DIM, DIM, |i, j| pseudo_random(i, j + 50)));
    let factor = pz.ids()[0];

    criterion.bench_function("compact", |b| b.iter(|| black_box(pz).compact()));
    criterion.bench_function("split", |b| b.iter(|| black_box(pz).split(factor)));
    criterion.bench_function("exact_plus", |b| {
        b.iter(|| black_box(pz).exact_plus(black_box(&other)));
    });
    criterion.bench_function("minkowski_sum", |b| {
        b.iter(|| black_box(pz).minkowski_sum(black_box(&other)));
    });
}

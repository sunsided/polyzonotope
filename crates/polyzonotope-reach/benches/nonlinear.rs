//! Benchmarks for nonlinear reachability of Van der Pol from a small box.

use std::hint::black_box;

use criterion::Criterion;
use nalgebra::{DMatrix, DVector};
use polyzonotope::{SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{NonlinearReach, NonlinearSettings, Polynomial};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    let vdp = Polynomial::new(2, 0)
        .term(0, 1.0, &[0, 1])
        .term(1, 1.0, &[0, 1])
        .term(1, -1.0, &[2, 1])
        .term(1, -1.0, &[1, 0]);
    let reach = NonlinearReach::new(vdp, NonlinearSettings::new(0.02));
    let start = SparsePolyZonotope::from_zonotope(&Zonotope::new(
        DVector::from_vec(vec![1.4, 2.4]),
        DMatrix::from_diagonal_element(2, 2, 0.05),
    ));
    let no_input = Zonotope::new(DVector::zeros(0), DMatrix::zeros(0, 0));

    criterion.bench_function("van_der_pol/step", |b| {
        b.iter(|| reach.step(black_box(&start), &no_input));
    });
    criterion.bench_function("van_der_pol/reach_50", |b| {
        b.iter(|| reach.reach(black_box(&start), &no_input, 50));
    });
    criterion.final_summary();
}

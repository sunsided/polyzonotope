//! Benchmarks for nonlinear reachability of Van der Pol from a small box.

use std::hint::black_box;

use criterion::Criterion;
use nalgebra::{DMatrix, DVector};
use polyzonotope::{SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{NonlinearReach, NonlinearSettings, Polynomial};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    let vdp = Polynomial::autonomous(2)
        .term(0, 1.0, &[0, 1])
        .term(1, 1.0, &[0, 1])
        .term(1, -1.0, &[2, 1])
        .term(1, -1.0, &[1, 0]);
    let reach = NonlinearReach::new(vdp, NonlinearSettings::new(0.02));
    let start = SparsePolyZonotope::from_zonotope(&Zonotope::new(
        DVector::from_vec(vec![1.4, 2.4]),
        DMatrix::from_diagonal_element(2, 2, 0.05),
    ));

    criterion.bench_function("van_der_pol/step", |b| {
        b.iter(|| reach.step_autonomous(black_box(&start)));
    });
    criterion.bench_function("van_der_pol/reach_50", |b| {
        b.iter(|| reach.reach_autonomous(black_box(&start), 50));
    });
    criterion.final_summary();
}

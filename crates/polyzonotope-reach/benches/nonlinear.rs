//! Benchmarks for nonlinear reachability of Van der Pol from a small box,
//! with both abstraction methods, and from a wide box that needs splitting.

use std::hint::black_box;

use criterion::Criterion;
use nalgebra::{DMatrix, DVector};
use polyzonotope::{SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{Method, NonlinearReach, NonlinearSettings, Polynomial};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    let vdp = Polynomial::autonomous(2)
        .term(0, 1.0, &[0, 1])
        .term(1, 1.0, &[0, 1])
        .term(1, -1.0, &[2, 1])
        .term(1, -1.0, &[1, 0]);
    let start = |radius| {
        SparsePolyZonotope::from_zonotope(&Zonotope::new(
            DVector::from_vec(vec![1.4, 2.4]),
            DMatrix::from_diagonal_element(2, 2, radius),
        ))
    };
    let (small, wide) = (start(0.05), start(1.0));

    for (name, method) in [
        ("linearization", Method::Linearization),
        ("polynomialization", Method::Polynomialization),
    ] {
        let settings = NonlinearSettings {
            method,
            ..NonlinearSettings::new(0.02)
        };
        let reach = NonlinearReach::new(vdp.clone(), settings);
        criterion.bench_function(&format!("van_der_pol/{name}/step"), |b| {
            b.iter(|| reach.step_autonomous(black_box(&small)));
        });
        criterion.bench_function(&format!("van_der_pol/{name}/reach_50"), |b| {
            b.iter(|| reach.reach_autonomous(black_box(&small), 50));
        });
        // Radius 1 diverges within a few steps; this times the splits and
        // merges on the way there.
        criterion.bench_function(&format!("van_der_pol/{name}/split_10"), |b| {
            b.iter(|| reach.reach_autonomous(black_box(&wide), 10));
        });
    }
    criterion.final_summary();
}

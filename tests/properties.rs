//! Soundness and exactness properties, checked on random sets and random samples.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, SparsePolyZonotope};
use proptest::collection::vec;
use proptest::prelude::*;

const TOL: f64 = 1e-9;

fn arb_pz() -> impl Strategy<Value = SparsePolyZonotope> {
    (1usize..=3, 0usize..=5, 0usize..=2, 1usize..=3).prop_flat_map(|(n, h, q, p)| {
        (
            vec(-5.0..5.0f64, n),
            vec(-2.0..2.0f64, n * h),
            vec(-2.0..2.0f64, n * q),
            vec(0u32..=3, p * h),
        )
            .prop_map(move |(c, g, gi, e)| {
                SparsePolyZonotope::new(
                    DVector::from_vec(c),
                    DMatrix::from_vec(n, h, g),
                    DMatrix::from_vec(n, q, gi),
                    DMatrix::from_vec(p, h, e),
                    FactorId::fresh_n(p),
                )
                .unwrap()
            })
    })
}

/// Random factor values: an α for every id (via a hash map) and enough β's.
#[derive(Clone, Debug)]
struct Sample {
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

fn arb_samples() -> impl Strategy<Value = Vec<Sample>> {
    vec(
        (vec(-1.0..=1.0f64, 8), vec(-1.0..=1.0f64, 8))
            .prop_map(|(alpha, beta)| Sample { alpha, beta }),
        1..20,
    )
}

fn alpha_map(ids: &[FactorId], s: &Sample) -> HashMap<FactorId, f64> {
    ids.iter().zip(&s.alpha).map(|(&k, &a)| (k, a)).collect()
}

fn eval(pz: &SparsePolyZonotope, s: &Sample) -> DVector<f64> {
    let a = alpha_map(pz.ids(), s);
    pz.evaluate_with(|k| a[&k], &s.beta[..pz.independent_generators().ncols()])
}

fn close(a: &DVector<f64>, b: &DVector<f64>) -> bool {
    (a - b).amax() <= TOL * (1.0 + a.amax())
}

proptest! {
    #[test]
    fn interval_hull_contains_every_point(pz in arb_pz(), samples in arb_samples()) {
        let hull = pz.interval_hull();
        for s in &samples {
            prop_assert!(hull.contains(&eval(&pz, s), TOL));
        }
    }

    /// Soundness *by construction*: for every point of the PZ we exhibit the
    /// zonotope coefficients that reproduce it, and check they lie in [-1, 1].
    #[test]
    fn to_zonotope_reproduces_every_point(pz in arb_pz(), samples in arb_samples()) {
        let z = pz.to_zonotope();
        let (e, ids) = (pz.exponents(), pz.ids());
        for s in &samples {
            let a = alpha_map(ids, s);
            let mut coeffs = Vec::new();
            for j in 0..e.ncols() {
                let col: Vec<u32> = e.column(j).iter().copied().collect();
                let m: f64 = col.iter().zip(ids).map(|(&x, k)| a[k].powi(x as i32)).product();
                if col.iter().all(|&x| x == 0) {
                    continue; // folded into the center
                } else if col.iter().all(|&x| x % 2 == 0) {
                    coeffs.push(2.0 * m - 1.0);
                } else {
                    coeffs.push(m);
                }
            }
            coeffs.extend_from_slice(&s.beta[..pz.independent_generators().ncols()]);
            prop_assert!(coeffs.iter().all(|c| c.abs() <= 1.0 + TOL));
            prop_assert!(close(&z.evaluate(&coeffs), &eval(&pz, s)));
        }
    }

    #[test]
    fn linear_map_is_exact(pz in arb_pz(), m in vec(-3.0..3.0f64, 9), samples in arb_samples()) {
        let n = pz.dim();
        let m = DMatrix::from_fn(n, n, |i, j| m[i * 3 + j]);
        let mapped = pz.linear_map(&m);
        for s in &samples {
            prop_assert!(close(&eval(&mapped, s), &(&m * eval(&pz, s))));
        }
    }

    #[test]
    fn compact_preserves_the_function(pz in arb_pz(), samples in arb_samples()) {
        let compacted = pz.compact();
        for s in &samples {
            // compact() keeps the relative order of ids, so look values up by id.
            let a = alpha_map(pz.ids(), s);
            let q = pz.independent_generators().ncols();
            let qc = compacted.independent_generators().ncols();
            let beta: Vec<f64> = pz.independent_generators().column_iter().zip(&s.beta)
                .filter(|(col, _)| col.iter().any(|&x| x != 0.0)).map(|(_, &b)| b).collect();
            prop_assert_eq!(beta.len(), qc);
            prop_assert!(close(&compacted.evaluate_with(|k| a[&k], &beta), &pz.evaluate_with(|k| a[&k], &s.beta[..q])));
        }
    }

    /// Shared factors stay correlated: (p + Mp)(α) = p(α) + M p(α).
    #[test]
    fn exact_plus_respects_shared_factors(pz in arb_pz(), samples in arb_samples()) {
        let n = pz.dim();
        let m = DMatrix::from_fn(n, n, |i, j| if i == j { 2.0 } else { 0.5 });
        let other = pz.linear_map(&m);
        let sum = pz.exact_plus(&other);
        let q = pz.independent_generators().ncols();
        for s in &samples {
            let a = alpha_map(pz.ids(), s);
            let beta = &s.beta[..q];
            let expected = pz.evaluate_with(|k| a[&k], beta) + other.evaluate_with(|k| a[&k], beta);
            // Independent generators are concatenated, so feed the same β twice.
            let beta2: Vec<f64> = beta.iter().chain(beta).copied().collect();
            let beta2: Vec<f64> = sum_beta(&pz, &other, &beta2);
            prop_assert!(close(&sum.evaluate_with(|k| a[&k], &beta2), &expected));
        }
    }

    #[test]
    fn split_halves_are_exact(pz in arb_pz(), samples in arb_samples()) {
        let k = pz.ids()[0];
        let (lo, hi) = pz.split(k);
        for s in &samples {
            let a = alpha_map(pz.ids(), s);
            let t = s.alpha[7]; // the rescaled factor α'
            let q = pz.independent_generators().ncols();
            for (half, sign) in [(&lo, -1.0), (&hi, 1.0)] {
                let new_id = half.ids().iter().copied().find(|id| !pz.ids().contains(id));
                let value = |id: FactorId| if Some(id) == new_id { t } else { a[&id] };
                let orig = |id: FactorId| if id == k { (t + sign) / 2.0 } else { a[&id] };
                let beta = &s.beta[..q];
                prop_assert!(close(&half.evaluate_with(value, &beta[..half.independent_generators().ncols()]), &pz.evaluate_with(orig, beta)));
            }
        }
    }
}

/// `exact_plus` compacts its result, which drops zero independent generators;
/// keep the β's of the surviving columns.
fn sum_beta(a: &SparsePolyZonotope, b: &SparsePolyZonotope, beta: &[f64]) -> Vec<f64> {
    a.independent_generators()
        .column_iter()
        .chain(b.independent_generators().column_iter())
        .zip(beta)
        .filter(|(col, _)| col.iter().any(|&x| x != 0.0))
        .map(|(_, &x)| x)
        .collect()
}

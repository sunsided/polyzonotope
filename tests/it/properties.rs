//! Soundness and exactness properties, checked on random sets and random samples.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, SparsePolyZonotope, Zonotope};
use proptest::collection::vec;
use proptest::prelude::*;

use crate::strategies::{Sample, TOL, alpha_map, arb_pz, arb_samples, close, eval};

proptest! {
    /// Every sampled point of a set lies in its interval hull.
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
        for s in &samples {
            let coeffs = zonotope_coefficients(&pz, s);

            prop_assert!(coeffs.iter().all(|c| c.abs() <= 1.0 + TOL));
            prop_assert!(close(&z.evaluate(&coeffs), &eval(&pz, s)));
        }
    }

    /// The zonotope enclosure loses nothing on the box level: its interval
    /// hull equals the polynomial zonotope's.
    #[test]
    fn to_zonotope_keeps_the_interval_hull(pz in arb_pz()) {
        let direct = pz.interval_hull();
        let via_zonotope = pz.to_zonotope().interval_hull();
        for (a, b) in direct.0.iter().zip(&via_zonotope.0) {
            prop_assert!((a.lo - b.lo).abs() <= TOL * (1.0 + a.lo.abs()));
            prop_assert!((a.hi - b.hi).abs() <= TOL * (1.0 + a.hi.abs()));
        }
    }

    /// Every sampled point of a zonotope lies in its interval hull.
    #[test]
    fn zonotope_hull_contains_every_point(pz in arb_pz(), samples in arb_samples()) {
        let z = pz.to_zonotope();
        let hull = z.interval_hull();
        for s in &samples {
            let beta: Vec<f64> = (0..z.num_generators()).map(|j| s.beta[j % s.beta.len()]).collect();
            prop_assert!(hull.contains(&z.evaluate(&beta), TOL));
        }
    }

    /// A box survives the round trip through [`Zonotope::from_box`] unchanged.
    #[test]
    fn from_box_round_trips_the_hull(pz in arb_pz()) {
        let hull = pz.interval_hull();
        let round_trip = Zonotope::from_box(&hull).interval_hull();
        for (a, b) in hull.0.iter().zip(&round_trip.0) {
            prop_assert!((a.lo - b.lo).abs() <= TOL * (1.0 + a.lo.abs()));
            prop_assert!((a.hi - b.hi).abs() <= TOL * (1.0 + a.hi.abs()));
        }
    }

    /// Linear maps commute with evaluation: `(M PZ)(α, β) = M PZ(α, β)`.
    #[test]
    fn linear_map_is_exact(pz in arb_pz(), m in vec(-3.0..3.0f64, 9), samples in arb_samples()) {
        let n = pz.dim();
        let m = DMatrix::from_fn(n, n, |i, j| m[i * 3 + j]);
        let mapped = pz.linear_map(&m);
        for s in &samples {
            prop_assert!(close(&eval(&mapped, s), &(&m * eval(&pz, s))));
        }
    }

    /// Translation commutes with evaluation: `(PZ + v)(α, β) = PZ(α, β) + v`.
    #[test]
    fn translate_is_exact(pz in arb_pz(), v in vec(-5.0..5.0f64, 3), samples in arb_samples()) {
        let v = DVector::from_column_slice(&v[..pz.dim()]);
        let moved = pz.translate(&v);
        for s in &samples {
            prop_assert!(close(&eval(&moved, s), &(eval(&pz, s) + &v)));
        }
    }

    /// Positional evaluation agrees with evaluation by id.
    #[test]
    fn evaluate_matches_evaluate_with(pz in arb_pz(), samples in arb_samples()) {
        let (p, q) = (pz.ids().len(), pz.independent_generators().ncols());
        for s in &samples {
            prop_assert!(close(&pz.evaluate(&s.alpha[..p], &s.beta[..q]), &eval(&pz, s)));
        }
    }

    /// Compacting changes the representation, not the function of α and β.
    #[test]
    fn compact_preserves_the_function(pz in arb_pz(), samples in arb_samples()) {
        let compacted = pz.compact();
        for s in &samples {
            // compact() keeps the relative order of ids, so look values up by id.
            let a = alpha_map(pz.ids(), s);
            let q = pz.independent_generators().ncols();
            let beta = &s.beta[..q];

            prop_assert_eq!(q, compacted.independent_generators().ncols());
            prop_assert!(close(&compacted.evaluate_with(|k| a[&k], beta), &pz.evaluate_with(|k| a[&k], beta)));
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
            let twice: Vec<f64> = beta.iter().chain(beta).copied().collect();
            prop_assert!(close(&sum.evaluate_with(|k| a[&k], &twice), &expected));
        }
    }

    /// The Minkowski sum of a set with itself decorrelates the two copies:
    /// `(p ⊕ p)(α₁, α₂) = p(α₁) + p(α₂)` for independent `α₁`, `α₂`.
    #[test]
    fn minkowski_sum_decorrelates_the_operands(pz in arb_pz(), s1 in arb_samples(), s2 in arb_samples()) {
        let sum = pz.minkowski_sum(&pz);
        // The second copy got fresh ids; compaction keeps the same factors of both copies, in order.
        let (old, new): (Vec<FactorId>, Vec<FactorId>) =
            sum.ids().iter().copied().partition(|id| pz.ids().contains(id));
        prop_assert_eq!(old.len(), new.len());

        let q = pz.independent_generators().ncols();
        for (s1, s2) in s1.iter().zip(&s2) {
            let (a1, a2) = (alpha_map(pz.ids(), s1), alpha_map(pz.ids(), s2));
            let value = |id: FactorId| match new.iter().position(|&n| n == id) {
                Some(j) => a2[&old[j]],
                None => a1[&id],
            };
            let beta: Vec<f64> = s1.beta[..q].iter().chain(&s2.beta[..q]).copied().collect();
            prop_assert!(close(&sum.evaluate_with(value, &beta), &(eval(&pz, s1) + eval(&pz, s2))));
        }
    }

    /// The Cartesian product of a set with its own image stacks both points
    /// for the same α: `(p × Mp)(α) = (p(α), M p(α))`.
    #[test]
    fn cartesian_product_keeps_shared_factors(pz in arb_pz(), samples in arb_samples()) {
        let n = pz.dim();
        let m = DMatrix::from_fn(n, n, |i, j| if i == j { -1.0 } else { 0.25 });
        let other = pz.linear_map(&m);
        let product = pz.cartesian_product(&other);
        let q = pz.independent_generators().ncols();
        for s in &samples {
            let a = alpha_map(pz.ids(), s);
            let beta = &s.beta[..q];
            let top = pz.evaluate_with(|k| a[&k], beta);
            let bottom = other.evaluate_with(|k| a[&k], beta);
            let expected = DVector::from_iterator(2 * n, top.iter().chain(bottom.iter()).copied());

            let twice: Vec<f64> = beta.iter().chain(beta).copied().collect();
            prop_assert!(close(&product.evaluate_with(|k| a[&k], &twice), &expected));
        }
    }

    /// Each half of a split reproduces the original set on its half of the
    /// factor's domain, via the rescaling `α = (α' ∓ 1) / 2`.
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
                let orig = |id: FactorId| if id == k { f64::midpoint(t, sign) } else { a[&id] };
                let beta = &s.beta[..q];
                prop_assert!(close(&half.evaluate_with(value, beta), &pz.evaluate_with(orig, beta)));
            }
        }
    }
}

/// The coefficients that make `pz.to_zonotope()` hit the point `eval(pz, s)`.
///
/// Mirrors the generator order of [`SparsePolyZonotope::to_zonotope`]: constant
/// monomials fold into the center, even monomials `m ∈ [0, 1]` become `2m - 1`,
/// the rest keep `m`, and the independent factors follow.
fn zonotope_coefficients(pz: &SparsePolyZonotope, s: &Sample) -> Vec<f64> {
    let (e, ids) = (pz.exponents(), pz.ids());
    let a = alpha_map(ids, s);
    let mut coeffs = Vec::new();
    for col in e.column_iter() {
        let m = monomial(col.iter().copied(), ids, &a);
        if col.iter().all(|&x| x == 0) {
            continue;
        }

        let even = col.iter().all(|&x| x % 2 == 0);
        coeffs.push(if even { 2.0 * m - 1.0 } else { m });
    }

    coeffs.extend_from_slice(&s.beta[..pz.independent_generators().ncols()]);
    coeffs
}

/// The monomial `Π_k α_k^exponents[k]` at the factor values `alpha`.
fn monomial(
    exponents: impl Iterator<Item = u32>,
    ids: &[FactorId],
    alpha: &HashMap<FactorId, f64>,
) -> f64 {
    exponents
        .zip(ids)
        .map(|(x, k)| alpha[k].powi(i32::try_from(x).expect("exponents are small")))
        .product()
}

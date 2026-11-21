//! Linear reachability against reference trajectories: every simulated state
//! lies in the reachable set at the end of the step and at every sampled time
//! inside it, and without inputs the time-point set is exactly `e^{Ar} X`.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{LinearReach, LinearSystem, ReachError, Step};
use proptest::collection::vec;
use proptest::prelude::*;

/// Relative tolerance for containment, covering the reference solvers' error.
const TOL: f64 = 1e-7;

/// A random linear system with `n = 2` or `3` states and one or two inputs.
fn arb_system() -> impl Strategy<Value = LinearSystem> {
    (2usize..=3, 1usize..=2).prop_flat_map(|(n, m)| {
        (vec(-2.0..2.0f64, n * n), vec(-1.0..1.0f64, n * m)).prop_map(move |(a, b)| LinearSystem {
            a: DMatrix::from_vec(n, n, a),
            b: DMatrix::from_vec(n, m, b),
        })
    })
}

/// A random initial set: a box-like zonotope with a curved term `α₀²`, with
/// generators scaled by `spread`. A point (`spread = 0`) leaves the hull
/// between the step ends no slack, so the curvature corrections must carry it.
fn initial_set(n: usize, values: &[f64], spread: f64) -> SparsePolyZonotope {
    let g = DMatrix::from_fn(n, n + 1, |i, j| {
        spread * values[(i * 7 + j * 3) % values.len()]
    });
    let mut e = DMatrix::from_fn(n, n + 1, |k, j| u32::from(k == j));
    e[(0, n)] = 2;
    SparsePolyZonotope::new(
        DVector::from_fn(n, |i, _| values[i]),
        g,
        DMatrix::zeros(n, 0),
        e,
        FactorId::fresh_n(n),
    )
    .expect("consistent shapes")
}

/// The input set `c_u ⊕ G_u` for `m` inputs.
fn input_set(m: usize, values: &[f64]) -> Zonotope {
    Zonotope::new(
        DVector::from_fn(m, |i, _| values[i]),
        DMatrix::from_fn(m, m, |i, j| {
            if i == j {
                0.2 * values[m + i].abs()
            } else {
                0.0
            }
        }),
    )
}

/// `x(t)` for `x' = A x + B u` with a constant `u`, from the exponential of
/// the augmented matrix `[[A, B u], [0, 0]]`.
fn exact(sys: &LinearSystem, x0: &DVector<f64>, u: &DVector<f64>, t: f64) -> DVector<f64> {
    let n = x0.len();
    let bu = &sys.b * u;
    let aug = DMatrix::from_fn(n + 1, n + 1, |i, j| match (i < n, j < n) {
        (true, true) => sys.a[(i, j)] * t,
        (true, false) => bu[i] * t,
        _ => 0.0,
    });
    let x = aug.exp() * DVector::from_fn(n + 1, |i, _| if i < n { x0[i] } else { 1.0 });
    x.rows(0, n).into_owned()
}

/// The point of `pz` for factor values drawn from `draws`, cycling.
fn point(pz: &SparsePolyZonotope, draws: &[f64]) -> DVector<f64> {
    let alpha: Vec<f64> = (0..pz.ids().len())
        .map(|k| draws[k % draws.len()])
        .collect();
    let beta: Vec<f64> = (0..pz.independent_generators().ncols())
        .map(|k| draws[(k + 3) % draws.len()])
        .collect();
    pz.evaluate(&alpha, &beta)
}

/// Whether `x` lies in the set, as seen by its interval hull and by the
/// support function of its zonotope enclosure in the given directions,
/// both ways along each.
fn inside(set: &SparsePolyZonotope, x: &DVector<f64>, directions: &[DVector<f64>]) -> bool {
    let slack = TOL * (1.0 + x.amax());
    let z = set.to_zonotope();
    let support = |d: &DVector<f64>| {
        d.dot(z.center())
            + z.generators()
                .column_iter()
                .map(|g| d.dot(&g).abs())
                .sum::<f64>()
    };

    set.interval_hull().contains(x, slack)
        && directions.iter().all(|d| {
            let margin = slack * (1.0 + d.amax());
            d.dot(x) <= support(d) + margin && -d.dot(x) <= support(&-d) + margin
        })
}

/// `‖A‖∞ = max_i Σ_j |A_ij|`.
fn infinity_norm(a: &DMatrix<f64>) -> f64 {
    a.row_iter().map(|row| row.abs().sum()).fold(0.0, f64::max)
}

/// The step, or `None` if the remainder bound rejects this step size.
fn step(
    sys: &LinearSystem,
    r: f64,
    terms: usize,
    x: &SparsePolyZonotope,
    u: &Zonotope,
) -> Option<Step> {
    match LinearReach::new(sys.clone(), r, terms) {
        Ok(reach) => Some(reach.step(x, u)),
        Err(ReachError::StepTooLarge) => None,
        Err(ReachError::InvalidStep) => unreachable!("test steps are positive"),
    }
}

/// A negative, infinite or NaN step is refused instead of giving a remainder
/// bound that is too small or panicking.
#[test]
fn invalid_steps_are_refused() {
    let sys = LinearSystem {
        a: DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]),
        b: DMatrix::from_element(2, 1, 1.0),
    };
    for r in [-0.1, -f64::MIN_POSITIVE, f64::INFINITY, f64::NAN] {
        assert_eq!(
            LinearReach::new(sys.clone(), r, 3).err(),
            Some(ReachError::InvalidStep),
            "r = {r}"
        );
    }
    assert!(LinearReach::new(sys, 0.0, 3).is_ok());
}

proptest! {
    /// With a constant input from the input set, the state at the end of the
    /// step and at sampled times inside it lies in the reachable sets.
    #[test]
    fn constant_inputs_stay_inside(
        sys in arb_system(),
        r in 0.02..0.6f64,
        terms in 2usize..=6,
        values in vec(-1.0..1.0f64, 8),
        spread in prop_oneof![Just(0.0), Just(0.3)],
        point_input in any::<bool>(),
        draws in vec(vec(-1.0..=1.0f64, 6), 1..6),
        directions in vec(vec(-1.0..1.0f64, 3), 4),
    ) {
        let (n, m) = (sys.a.nrows(), sys.b.ncols());
        // Keep ‖A‖∞ r <= 0.6: with η = 2 a larger product inflates E(r) to
        // tens of units and every state passes regardless of F, F̃ and R_p.
        let r = r.min(0.6 / infinity_norm(&sys.a));
        let x0 = initial_set(n, &values, spread);
        let u = input_set(m, &values);
        // A constant input without spread exposes the input correction F̃ the same way.
        let u = if point_input { Zonotope::new(u.center().clone(), DMatrix::zeros(m, 0)) } else { u };
        let Some(Step { time_point, time_interval }) = step(&sys, r, terms, &x0, &u) else {
            return Ok(());
        };
        let dirs: Vec<DVector<f64>> = directions.iter().map(|d| DVector::from_column_slice(&d[..n])).collect();
        for d in &draws {
            let start = point(&x0, d);
            let input = u.evaluate(&d[..u.num_generators()]);
            prop_assert!(inside(&time_point, &exact(&sys, &start, &input, r), &dirs));
            for k in 0..=8 {
                let t = r * f64::from(k) / 8.0;
                prop_assert!(inside(&time_interval, &exact(&sys, &start, &input, t), &dirs), "t = {t}");
            }
        }
    }

    /// Without inputs the time-point set is `e^{Ar} X` itself: at the factor
    /// values of a start point it gives that point's successor, up to the
    /// tiny Taylor remainder and with the added generators at zero.
    #[test]
    fn time_point_is_exact_without_inputs(
        sys in arb_system(),
        r in 0.01..0.2f64,
        values in vec(-1.0..1.0f64, 8),
        draws in vec(vec(-1.0..=1.0f64, 6), 1..6),
    ) {
        let (n, m) = (sys.a.nrows(), sys.b.ncols());
        let x0 = initial_set(n, &values, 0.3);
        let still = Zonotope::new(DVector::zeros(m), DMatrix::zeros(m, 0));
        let Some(step) = step(&sys, r, 12, &x0, &still) else { return Ok(()); };
        let p = &step.time_point;
        for d in &draws {
            let alpha: Vec<f64> = (0..x0.ids().len()).map(|k| d[k % d.len()]).collect();
            let by_id = |id: FactorId| x0.ids().iter().position(|&k| k == id).map_or(0.0, |k| alpha[k]);
            let got = p.evaluate_with(by_id, &vec![0.0; p.independent_generators().ncols()]);
            let want = exact(&sys, &x0.evaluate(&alpha, &[]), &DVector::zeros(m), r);
            prop_assert!((&got - &want).amax() <= 1e-9 * (1.0 + want.amax()), "{got} vs {want}");
        }
    }
}

proptest! {
    /// From rest at the origin under a constant input, the state traces the
    /// curve `Γ(t) B u`, off the chord between the step ends. With many Taylor
    /// terms the remainder is negligible, so the input correction `F̃` alone
    /// must cover the gap.
    #[test]
    fn input_curve_stays_inside(
        sys in arb_system(),
        r in 0.1..0.6f64,
        values in vec(-1.0..1.0f64, 8),
        directions in vec(vec(-1.0..1.0f64, 3), 4),
    ) {
        let (n, m) = (sys.a.nrows(), sys.b.ncols());
        let origin = initial_set(n, &[0.0; 8], 0.0);
        let u = Zonotope::new(DVector::from_fn(m, |i, _| values[i]), DMatrix::zeros(m, 0));
        let Some(step) = step(&sys, r, 10, &origin, &u) else { return Ok(()); };
        let dirs: Vec<DVector<f64>> = directions.iter().map(|d| DVector::from_column_slice(&d[..n])).collect();
        for k in 0..=16 {
            let t = r * f64::from(k) / 16.0;
            let x = exact(&sys, &DVector::zeros(n), u.center(), t);
            prop_assert!(inside(&step.time_interval, &x, &dirs), "t = {t}");
        }
    }
}

proptest! {
    /// An input that jumps between values of the input set during the step,
    /// solved exactly segment by segment, keeps the state inside both sets.
    #[test]
    fn varying_inputs_stay_inside(
        sys in arb_system(),
        r in 0.05..0.5f64,
        values in vec(-1.0..1.0f64, 8),
        switches in vec(vec(-1.0..=1.0f64, 2), 2..8),
        start in vec(-1.0..=1.0f64, 6),
        directions in vec(vec(-1.0..1.0f64, 3), 4),
    ) {
        let (n, m) = (sys.a.nrows(), sys.b.ncols());
        let (x0, u) = (initial_set(n, &values, 0.3), input_set(m, &values));
        let Some(step) = step(&sys, r, 6, &x0, &u) else { return Ok(()); };
        let dirs: Vec<DVector<f64>> = directions.iter().map(|d| DVector::from_column_slice(&d[..n])).collect();
        let dt = r / f64::from(u32::try_from(switches.len()).expect("few switches"));
        let mut x = point(&x0, &start);
        for s in &switches {
            x = exact(&sys, &x, &u.evaluate(&s[..m]), dt);
            prop_assert!(inside(&step.time_interval, &x, &dirs));
        }
        prop_assert!(inside(&step.time_point, &x, &dirs));
    }
}

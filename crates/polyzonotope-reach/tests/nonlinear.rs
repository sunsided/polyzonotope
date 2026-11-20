//! Nonlinear reachability: linear dynamics give the sets of the linear
//! reach, and trajectories of Van der Pol, a pendulum, a cubic drift and a
//! bilinear input system, integrated with a fine RK4, stay inside the sets
//! at every step and every sampled time, for both abstraction methods.

use core::f64::consts::{FRAC_PI_2, TAU};

use nalgebra::{DMatrix, DVector};
use polyzonotope::{Interval, IntervalBox, IntervalMatrix, SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{
    Dynamics, IntervalTensor, LinearReach, LinearSystem, Method, NonlinearReach, NonlinearSettings,
    Polynomial, ReachError, Splitting, Step,
};
use proptest::collection::vec;
use proptest::prelude::*;

/// Absolute tolerance for containment, covering the RK4 reference's error.
const TOL: f64 = 1e-7;

/// RK4 sub-steps per reachability step for the reference trajectories.
const SUBSTEPS: u32 = 40;

/// Van der Pol, `x₀' = x₁`, `x₁' = μ (1 − x₀²) x₁ − x₀ + u`, with `μ = 1`
/// and one additive input.
fn van_der_pol() -> Polynomial {
    Polynomial::new(2, 1)
        .term(0, 1.0, &[0, 1, 0])
        .term(1, 1.0, &[0, 1, 0])
        .term(1, -1.0, &[2, 1, 0])
        .term(1, -1.0, &[1, 0, 0])
        .term(1, 1.0, &[0, 0, 1])
}

/// The vector field of [`van_der_pol`] in plain `f64`.
fn van_der_pol_f64(x: &DVector<f64>, u: f64) -> DVector<f64> {
    DVector::from_vec(vec![x[1], (1.0 - x[0] * x[0]) * x[1] - x[0] + u])
}

/// The pendulum `θ' = ω`, `ω' = −sin θ + u` with a torque input, enclosed
/// with interval sine and cosine.
struct Pendulum;

impl Dynamics for Pendulum {
    fn states(&self) -> usize {
        2
    }

    fn inputs(&self) -> usize {
        1
    }

    fn flow(&self, z: &[Interval]) -> Vec<Interval> {
        vec![z[1], z[2] - sin(z[0])]
    }

    fn jacobian(&self, z: &[Interval]) -> IntervalMatrix {
        let mut j = IntervalMatrix::zeros(2, 3);
        j.0[(0, 1)] = Interval::point(1.0);
        j.0[(1, 0)] = -cos(z[0]);
        j.0[(1, 2)] = Interval::point(1.0);
        j
    }

    fn hessians(&self, z: &[Interval]) -> Vec<IntervalMatrix> {
        let mut h = IntervalMatrix::zeros(3, 3);
        h.0[(0, 0)] = sin(z[0]);
        vec![IntervalMatrix::zeros(3, 3), h]
    }

    fn third_derivatives(&self, z: &[Interval]) -> Option<Vec<IntervalTensor>> {
        let mut t = IntervalTensor::zeros(3);
        t[(0, 0, 0)] = cos(z[0]);
        Some(vec![IntervalTensor::zeros(3), t])
    }
}

/// The vector field of [`Pendulum`] in plain `f64`.
fn pendulum_f64(x: &DVector<f64>, u: f64) -> DVector<f64> {
    DVector::from_vec(vec![x[1], u - x[0].sin()])
}

/// `{ sin x | x ∈ iv }`: the end values, plus ±1 where the interval holds a
/// peak, widened by `1e-12` for libm's rounding.
fn sin(iv: Interval) -> Interval {
    if iv.width() >= TAU {
        return Interval::new(-1.0, 1.0);
    }

    let holds = |peak: f64| {
        let k = ((iv.lo - peak) / TAU).ceil();
        peak + k * TAU <= iv.hi
    };
    let (a, b) = (iv.lo.sin(), iv.hi.sin());
    let hi = if holds(FRAC_PI_2) { 1.0 } else { a.max(b) };
    let lo = if holds(-FRAC_PI_2) { -1.0 } else { a.min(b) };
    Interval::new(lo - 1e-12, hi + 1e-12)
}

/// `{ cos x | x ∈ iv }`, as a shifted [`sin`].
fn cos(iv: Interval) -> Interval {
    sin(Interval::new(
        iv.lo + FRAC_PI_2 - 1e-12,
        iv.hi + FRAC_PI_2 + 1e-12,
    ))
}

/// Settings with step `r`, abstraction `method` and enough Taylor terms
/// that the remainder of the matrix exponential is negligible, so that the
/// tests see the abstraction error and not the slack of the linear part.
fn sharp(r: f64, method: Method) -> NonlinearSettings {
    NonlinearSettings {
        taylor_terms: 12,
        method,
        ..NonlinearSettings::new(r)
    }
}

/// Runs the scenario `$scenario(method: Method)` as one test per method, in
/// a module `$name` with tests `linearization` and `polynomialization`.
macro_rules! both_methods {
    ($name:ident, $scenario:ident) => {
        /// The scenario of the same name, once per abstraction method.
        mod $name {
            use super::*;

            /// With conservative linearization.
            #[test]
            fn linearization() {
                $scenario(Method::Linearization);
            }

            /// With conservative polynomialization.
            #[test]
            fn polynomialization() {
                $scenario(Method::Polynomialization);
            }
        }
    };
}

/// The box `center ± radius` as a polynomial zonotope with dependent factors.
fn initial_box(center: &[f64], radius: f64) -> SparsePolyZonotope {
    SparsePolyZonotope::from_zonotope(&Zonotope::new(
        DVector::from_column_slice(center),
        DMatrix::from_diagonal_element(center.len(), center.len(), radius),
    ))
}

/// The input interval `[-radius, radius]`.
fn input(radius: f64) -> Zonotope {
    Zonotope::new(DVector::zeros(1), DMatrix::from_element(1, 1, radius))
}

/// A tiny linear congruential generator in `[-1, 1]`, so that the samples
/// are reproducible without a dependency.
struct Samples(u64);

impl Samples {
    /// The next value in `[-1, 1]`.
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[expect(clippy::cast_precision_loss, reason = "the top 53 bits fit exactly")]
        let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * unit - 1.0
    }
}

/// One RK4 step of length `h` with the input held at `u`.
fn rk4(
    f: fn(&DVector<f64>, f64) -> DVector<f64>,
    x: &DVector<f64>,
    u: f64,
    h: f64,
) -> DVector<f64> {
    let k1 = f(x, u);
    let k2 = f(&(x + &k1 * (h / 2.0)), u);
    let k3 = f(&(x + &k2 * (h / 2.0)), u);
    let k4 = f(&(x + &k3 * h), u);
    x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
}

/// A set's interval hull and zonotope enclosure, computed once and used as
/// a necessary test for membership.
struct Enclosure {
    /// The interval hull.
    hull: IntervalBox,
    /// The zonotope enclosure.
    zonotope: Zonotope,
}

impl Enclosure {
    /// The enclosures of `set`.
    fn of(set: &SparsePolyZonotope) -> Self {
        Self {
            hull: set.interval_hull(),
            zonotope: set.to_zonotope(),
        }
    }

    /// Whether `x` lies in the hull and below the zonotope's support
    /// function in 16 directions.
    fn holds(&self, x: &DVector<f64>) -> bool {
        let z = &self.zonotope;
        self.hull.contains(x, TOL)
            && (0..16).all(|k| {
                let angle = TAU * f64::from(k) / 16.0;
                let d = DVector::from_vec(vec![angle.cos(), angle.sin()]);
                let support = d.dot(z.center()) + (d.transpose() * z.generators()).abs().sum();
                d.dot(x) <= support + TOL
            })
    }
}

/// Simulates `samples` trajectories from corners and random points of
/// `center ± radius` and checks each against every step of `flowpipe`:
/// inside some piece's time-point set at the step end, and inside some
/// piece's time-interval set at every sub-step.
///
/// Every third sample holds the input at `u_radius`, every third at
/// `-u_radius`, and the rest jump inside `[-u_radius, u_radius]` every
/// sub-step. A jumping input averages out, so only the held ones reach the
/// far edges of the set.
fn check_trajectories(
    f: fn(&DVector<f64>, f64) -> DVector<f64>,
    flowpipe: &[Vec<Step>],
    (center, radius, u_radius): (&[f64], f64, f64),
    r: f64,
    samples: usize,
) {
    let enclosures: Vec<Vec<(Enclosure, Enclosure)>> = flowpipe
        .iter()
        .map(|pieces| {
            pieces
                .iter()
                .map(|p| {
                    (
                        Enclosure::of(&p.time_interval),
                        Enclosure::of(&p.time_point),
                    )
                })
                .collect()
        })
        .collect();

    let mut rng = Samples(7);
    for s in 0..samples {
        let mut x = DVector::from_fn(2, |i, _| {
            let corner = if (s >> i) & 1 == 0 { -1.0 } else { 1.0 };
            center[i] + radius * if s < 4 { corner } else { rng.next() }
        });
        for (k, pieces) in enclosures.iter().enumerate() {
            for _ in 0..SUBSTEPS {
                let u = match s % 3 {
                    0 => u_radius,
                    1 => -u_radius,
                    _ => u_radius * rng.next(),
                };
                x = rk4(f, &x, u, r / f64::from(SUBSTEPS));
                assert!(
                    pieces.iter().any(|(during, _)| during.holds(&x)),
                    "sample {s} left the time-interval set of step {k} at {x}"
                );
            }

            assert!(
                pieces.iter().any(|(_, end)| end.holds(&x)),
                "sample {s} left the time-point set of step {k} at {x}"
            );
        }
    }
}

/// Van der Pol from a box, with an input: every sampled trajectory stays in
/// the sets over 0.8 s.
fn van_der_pol_contains_trajectories(method: Method) {
    let (center, radius, u) = ([1.4, 2.4], 0.05, 0.05);
    let reach = NonlinearReach::new(van_der_pol(), sharp(0.02, method));
    let flowpipe = reach
        .reach(&initial_box(&center, radius), &input(u), 40)
        .expect("converges for this step");
    check_trajectories(van_der_pol_f64, &flowpipe, (&center, radius, u), 0.02, 24);
}

/// From a single point and without input, the abstraction error alone has
/// to carry the curvature of the trajectory.
fn van_der_pol_contains_a_point_trajectory(method: Method) {
    let reach = NonlinearReach::new(van_der_pol(), sharp(0.05, method));
    let flowpipe = reach
        .reach(&initial_box(&[1.4, 2.4], 0.0), &input(0.0), 40)
        .expect("converges for this step");
    check_trajectories(van_der_pol_f64, &flowpipe, (&[1.4, 2.4], 0.0, 0.0), 0.05, 1);
}

/// The pendulum at large angles, with a torque input: every sampled
/// trajectory stays in the sets over 1 s.
fn pendulum_contains_trajectories(method: Method) {
    let (center, radius, u) = ([1.2, 0.0], 0.05, 0.02);
    let reach = NonlinearReach::new(Pendulum, sharp(0.025, method));
    let flowpipe = reach
        .reach(&initial_box(&center, radius), &input(u), 40)
        .expect("converges for this step");
    check_trajectories(pendulum_f64, &flowpipe, (&center, radius, u), 0.025, 24);
}

/// `x₀' = x₀³`, `x₁' = −x₁` around the origin: the linearization there is
/// blind to the cubic, so only the Hessian's spread over the set, not its
/// value at the center, holds the outward drift.
fn cubic_drift_is_contained(method: Method) {
    /// The vector field in plain `f64`.
    fn cubic(x: &DVector<f64>, _: f64) -> DVector<f64> {
        DVector::from_vec(vec![x[0].powi(3), -x[1]])
    }

    let dynamics = Polynomial::new(2, 1)
        .term(0, 1.0, &[3, 0, 0])
        .term(1, -1.0, &[0, 1, 0]);
    let reach = NonlinearReach::new(dynamics, sharp(0.05, method));
    let flowpipe = reach
        .reach(&initial_box(&[0.0, 0.0], 0.5), &input(0.0), 10)
        .expect("converges for this step");
    check_trajectories(cubic, &flowpipe, (&[0.0, 0.0], 0.5, 0.0), 0.05, 24);
}

/// A set far too large for the step is refused, not enclosed loosely.
fn oversized_set_is_refused(method: Method) {
    let settings = NonlinearSettings {
        max_pieces: 1,
        method,
        ..NonlinearSettings::new(0.2)
    };
    let reach = NonlinearReach::new(van_der_pol(), settings);
    let result = reach.reach(&initial_box(&[1.4, 2.4], 3.0), &input(0.0), 5);
    assert_eq!(
        result.err().map(|f| f.error),
        Some(ReachError::AbstractionDiverged)
    );
}

/// A degree-9 field that overflows `f64` during the error check is
/// refused, not a panic, even under `sound`.
fn overflow_is_refused(method: Method) {
    let dynamics = Polynomial::autonomous(2)
        .term(0, 1.0, &[9, 0])
        .term(1, -1.0, &[1, 9]);
    let settings = NonlinearSettings {
        max_pieces: 4,
        max_iterations: 30,
        enlarge: 10.0,
        method,
        ..NonlinearSettings::new(0.05)
    };
    let result =
        NonlinearReach::new(dynamics, settings).reach_autonomous(&initial_box(&[0.0, 0.0], 1.0), 5);
    assert_eq!(
        result.err().map(|f| f.error),
        Some(ReachError::AbstractionDiverged)
    );
}

/// Splitting carries a large set through a step that one piece cannot,
/// and the pieces still contain every sampled trajectory.
#[test]
fn splitting_rescues_a_large_set() {
    let (center, radius, r, steps) = ([1.4, 2.4], 1.0, 0.02, 9);
    let start = initial_box(&center, radius);
    let single = NonlinearSettings {
        max_pieces: 1,
        ..sharp(r, Method::Linearization)
    };
    let unsplit =
        NonlinearReach::new(van_der_pol(), single.clone()).reach(&start, &input(0.0), steps);
    assert_eq!(
        unsplit.err().map(|f| f.error),
        Some(ReachError::AbstractionDiverged)
    );

    let split = NonlinearSettings {
        max_pieces: 8,
        ..single
    };
    let flowpipe = NonlinearReach::new(van_der_pol(), split)
        .reach(&start, &input(0.0), steps)
        .expect("splitting makes the step converge");
    assert!(flowpipe.last().expect("nine steps").len() > 1);
    check_trajectories(van_der_pol_f64, &flowpipe, (&center, radius, 0.0), r, 24);
}

both_methods!(van_der_pol_box, van_der_pol_contains_trajectories);
both_methods!(van_der_pol_point, van_der_pol_contains_a_point_trajectory);
both_methods!(pendulum, pendulum_contains_trajectories);
both_methods!(cubic_drift, cubic_drift_is_contained);
both_methods!(oversized, oversized_set_is_refused);
both_methods!(overflow, overflow_is_refused);
both_methods!(bilinear, bilinear_input_is_contained);
both_methods!(merging, pieces_merge_as_the_set_contracts);

/// `x₀' = −x₀ − x₀³`, `x₁' = −x₁ + ½ x₀²` from a wide box: the cubic forces
/// splits early on, and the contraction toward the origin later lets the
/// pieces merge again. The merged sets still hold every trajectory.
fn pieces_merge_as_the_set_contracts(method: Method) {
    /// The vector field in plain `f64`.
    fn damped(x: &DVector<f64>, _: f64) -> DVector<f64> {
        DVector::from_vec(vec![-x[0] - x[0].powi(3), -x[1] + 0.5 * x[0] * x[0]])
    }

    let dynamics = Polynomial::new(2, 1)
        .term(0, -1.0, &[1, 0, 0])
        .term(0, -1.0, &[3, 0, 0])
        .term(1, -1.0, &[0, 1, 0])
        .term(1, 0.5, &[2, 0, 0]);
    let flowpipe = NonlinearReach::new(dynamics, sharp(0.1, method))
        .reach(&initial_box(&[0.0, 0.0], 1.0), &input(0.0), 40)
        .expect("splitting carries the set");
    let counts: Vec<usize> = flowpipe.iter().map(Vec::len).collect();
    assert!(counts.iter().any(|&c| c > 1), "{counts:?}");
    assert!(counts.windows(2).any(|w| w[1] < w[0]), "{counts:?}");
    check_trajectories(damped, &flowpipe, (&[0.0, 0.0], 1.0, 0.0), 0.1, 12);
}

/// `x₀' = x₁`, `x₁' = −x₀ − x₀ u`: a spring whose stiffness the input
/// modulates. `∂²f/∂x∂u ≠ 0`, so the error has a part from the input
/// varying during the step, which additive inputs never exercise.
fn bilinear() -> Polynomial {
    Polynomial::new(2, 1)
        .term(0, 1.0, &[0, 1, 0])
        .term(1, -1.0, &[1, 0, 0])
        .term(1, -1.0, &[1, 0, 1])
}

/// The vector field of [`bilinear`] in plain `f64`.
fn bilinear_f64(x: &DVector<f64>, u: f64) -> DVector<f64> {
    DVector::from_vec(vec![x[1], -x[0] - x[0] * u])
}

/// The bilinear spring with a stiffness input jumping in `[-0.5, 0.5]`:
/// every sampled trajectory stays in the sets over 2 s.
fn bilinear_input_is_contained(method: Method) {
    let (center, radius, u) = ([1.0, 0.5], 0.1, 0.5);
    let reach = NonlinearReach::new(bilinear(), sharp(0.05, method));
    let flowpipe = reach
        .reach(&initial_box(&center, radius), &input(u), 40)
        .expect("converges for this step");
    check_trajectories(bilinear_f64, &flowpipe, (&center, radius, u), 0.05, 24);
}

/// From radius 0.5, Van der Pol defeats linearization within a second
/// without splitting, while polynomialization carries the set through and
/// still contains every sampled trajectory.
#[test]
fn polynomialization_carries_a_large_set() {
    let (center, radius, r, steps) = ([1.4, 2.4], 0.5, 0.02, 40);
    let start = initial_box(&center, radius);
    let settings = |method| NonlinearSettings {
        max_pieces: 1,
        ..sharp(r, method)
    };

    let linearized = NonlinearReach::new(van_der_pol(), settings(Method::Linearization)).reach(
        &start,
        &input(0.0),
        steps,
    );
    assert!(linearized.is_err());

    let flowpipe = NonlinearReach::new(van_der_pol(), settings(Method::Polynomialization))
        .reach(&start, &input(0.0), steps)
        .expect("polynomialization converges");
    check_trajectories(van_der_pol_f64, &flowpipe, (&center, radius, 0.0), r, 24);
}

/// On a set both methods carry, polynomialization's sets are smaller once
/// the set is wide enough for the quadratic error to dominate.
#[test]
fn polynomialization_is_tighter_on_a_wide_set() {
    let width = |flowpipe: &[Vec<Step>]| -> f64 {
        let hull = flowpipe.last().expect("steps")[0]
            .time_point
            .interval_hull();
        hull.0.iter().map(Interval::width).sum()
    };
    let start = initial_box(&[1.4, 2.4], 0.2);
    let run = |method| {
        NonlinearReach::new(van_der_pol(), sharp(0.02, method))
            .reach(&start, &input(0.0), 40)
            .expect("converges")
    };

    let linearized = width(&run(Method::Linearization));
    let polynomialized = width(&run(Method::Polynomialization));
    assert!(
        polynomialized < 0.9 * linearized,
        "{polynomialized} vs {linearized}"
    );
}

/// Dynamics without third derivatives are refused by polynomialization at
/// once, without spending pieces on splits, and still work with
/// linearization.
#[test]
fn missing_derivatives_are_refused() {
    /// `x' = −x³` without third derivatives.
    struct Cubic;

    impl Dynamics for Cubic {
        fn states(&self) -> usize {
            1
        }

        fn inputs(&self) -> usize {
            0
        }

        fn flow(&self, z: &[Interval]) -> Vec<Interval> {
            vec![-(z[0] * z[0] * z[0])]
        }

        fn jacobian(&self, z: &[Interval]) -> IntervalMatrix {
            IntervalMatrix(DMatrix::from_element(
                1,
                1,
                Interval::point(-3.0) * z[0] * z[0],
            ))
        }

        fn hessians(&self, z: &[Interval]) -> Vec<IntervalMatrix> {
            vec![IntervalMatrix(DMatrix::from_element(
                1,
                1,
                Interval::point(-6.0) * z[0],
            ))]
        }
    }

    let start = initial_box(&[1.0], 0.1);
    let failure = NonlinearReach::new(Cubic, sharp(0.05, Method::Polynomialization))
        .reach_autonomous(&start, 3)
        .expect_err("no third derivatives");
    assert_eq!(failure.error, ReachError::MissingDerivatives);
    assert_eq!(failure.proven, Vec::<Vec<Step>>::new());

    let flowpipe = NonlinearReach::new(Cubic, sharp(0.05, Method::Linearization))
        .reach_autonomous(&start, 3)
        .expect("linearization needs only the Hessians");
    assert_eq!(flowpipe.len(), 3);
}

/// The static error is quadratic in the set, so without the cap the degree
/// would double every step; over 60 steps it stays at `max_degree`.
#[test]
fn degree_stays_capped() {
    let settings = NonlinearSettings {
        max_degree: 3,
        ..sharp(0.02, Method::Polynomialization)
    };
    let flowpipe = NonlinearReach::new(van_der_pol(), settings)
        .reach(&initial_box(&[1.4, 2.4], 0.2), &input(0.05), 60)
        .expect("converges");
    for step in flowpipe.iter().flatten() {
        let e = step.time_point.exponents();
        assert!((0..e.ncols()).all(|j| e.column(j).sum() <= 3));
    }
}

/// Factor splits alone, which keep every dependency, carry the cubic drift
/// from radius 0.5, with more pieces than the default needs, and the pieces
/// contain every sampled trajectory.
#[test]
fn factor_splits_carry_the_cubic_drift() {
    /// The vector field in plain `f64`.
    fn cubic(x: &DVector<f64>, _: f64) -> DVector<f64> {
        DVector::from_vec(vec![x[0].powi(3), -x[1]])
    }

    let dynamics = Polynomial::new(2, 1)
        .term(0, 1.0, &[3, 0, 0])
        .term(1, -1.0, &[0, 1, 0]);
    let settings = NonlinearSettings {
        splitting: Splitting::Factors,
        max_pieces: 64,
        ..sharp(0.05, Method::Linearization)
    };
    let flowpipe = NonlinearReach::new(dynamics, settings)
        .reach(&initial_box(&[0.0, 0.0], 0.5), &input(0.0), 10)
        .expect("factor splits carry the set");
    assert!(flowpipe.last().expect("ten steps").len() > 8);
    check_trajectories(cubic, &flowpipe, (&[0.0, 0.0], 0.5, 0.0), 0.05, 24);
}

/// A random linear system `x' = A x + B u` with two states and one input,
/// as polynomial dynamics and as matrices.
fn arb_linear() -> impl Strategy<Value = (Polynomial, LinearSystem)> {
    (vec(-2.0..2.0f64, 4), vec(-1.0..1.0f64, 2)).prop_map(|(a, b)| {
        let mut p = Polynomial::new(2, 1);
        for i in 0..2 {
            p = p
                .term(i, a[2 * i], &[1, 0, 0])
                .term(i, a[2 * i + 1], &[0, 1, 0])
                .term(i, b[i], &[0, 0, 1]);
        }
        let system = LinearSystem {
            a: DMatrix::from_row_slice(2, 2, &a),
            b: DMatrix::from_column_slice(2, 1, &b),
        };
        (p, system)
    })
}

/// Whether two boxes agree up to `tol` per bound.
fn same_hull(a: &SparsePolyZonotope, b: &SparsePolyZonotope, tol: f64) -> bool {
    a.interval_hull()
        .0
        .iter()
        .zip(&b.interval_hull().0)
        .all(|(x, y)| (x.lo - y.lo).abs() <= tol && (x.hi - y.hi).abs() <= tol)
}

proptest! {
    // A fixed identity rather than a search: 64 cases are plenty, and the
    // default 256 (or test:prop's 10000) would cost minutes in debug builds.
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// For linear dynamics the abstraction error vanishes, so one nonlinear
    /// step gives the sets of the linear reach, up to rounding.
    #[test]
    fn linear_dynamics_match_linear_reach(
        (dynamics, system) in arb_linear(),
        center in vec(-1.0..1.0f64, 2),
        u_center in -1.0..1.0f64,
    ) {
        let r = 0.05;
        let x = initial_box(&center, 0.1);
        let u = Zonotope::new(DVector::from_element(1, u_center), DMatrix::from_element(1, 1, 0.1));
        let settings = NonlinearSettings { order: 1e6, ..NonlinearSettings::new(r) };

        let expected = LinearReach::new(system, r, settings.taylor_terms).expect("small step").step(&x, &u);
        let actual = NonlinearReach::new(dynamics, settings).step(&x, &u).expect("no error to bound");
        prop_assert!(same_hull(&actual.time_point, &expected.time_point, 1e-9));
        prop_assert!(same_hull(&actual.time_interval, &expected.time_interval, 1e-9));
    }

    /// The same for polynomialization, whose step runs on a system
    /// augmented with the static error: with no error, the augmentation
    /// must not change the sets. 12 Taylor terms, since the augmented `A`
    /// has a larger norm and so a larger remainder.
    #[test]
    fn linear_dynamics_match_linear_reach_polynomialized(
        (dynamics, system) in arb_linear(),
        center in vec(-1.0..1.0f64, 2),
        u_center in -1.0..1.0f64,
    ) {
        let r = 0.05;
        let x = initial_box(&center, 0.1);
        let u = Zonotope::new(DVector::from_element(1, u_center), DMatrix::from_element(1, 1, 0.1));
        let settings = NonlinearSettings { order: 1e6, ..sharp(r, Method::Polynomialization) };

        let expected = LinearReach::new(system, r, settings.taylor_terms).expect("small step").step(&x, &u);
        let actual = NonlinearReach::new(dynamics, settings).step(&x, &u).expect("no error to bound");
        prop_assert!(same_hull(&actual.time_point, &expected.time_point, 1e-9));
        prop_assert!(same_hull(&actual.time_interval, &expected.time_interval, 1e-9));
    }
}

/// A run that fails part-way hands back the steps it proved, and they
/// match the first steps of a run that stops before the failure.
#[test]
fn failure_keeps_the_proven_steps() {
    let settings = NonlinearSettings {
        max_pieces: 1,
        ..sharp(0.02, Method::Linearization)
    };
    let reach = NonlinearReach::new(van_der_pol(), settings);
    let start = initial_box(&[1.4, 2.4], 1.0);
    let failure = reach
        .reach(&start, &input(0.0), 9)
        .expect_err("diverges unsplit");
    assert_eq!(failure.error, ReachError::AbstractionDiverged);
    assert!(failure.proven.len() < 9);

    let shorter = reach
        .reach(&start, &input(0.0), failure.proven.len())
        .expect("the proven prefix succeeds");
    assert!(same_flowpipe(&shorter, &failure.proven));
}

/// The autonomous helpers give the same sets as an explicit empty input.
#[test]
fn autonomous_helpers_match_an_empty_input() {
    let vdp = Polynomial::autonomous(2)
        .term(0, 1.0, &[0, 1])
        .term(1, 1.0, &[0, 1])
        .term(1, -1.0, &[2, 1])
        .term(1, -1.0, &[1, 0]);
    let reach = NonlinearReach::new(vdp, sharp(0.02, Method::Linearization));
    let start = initial_box(&[1.4, 2.4], 0.05);
    let empty = Zonotope::new(DVector::zeros(0), DMatrix::zeros(0, 0));
    let helper = reach.reach_autonomous(&start, 3).expect("small set");
    let explicit = reach.reach(&start, &empty, 3).expect("small set");
    assert!(same_flowpipe(&helper, &explicit));
}

/// Whether two flowpipes have the same pieces, compared by their interval
/// hulls: factor ids are fresh on every run, so the sets never compare equal.
fn same_flowpipe(a: &[Vec<Step>], b: &[Vec<Step>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len()
                && x.iter().zip(y).all(|(p, q)| {
                    p.time_point.interval_hull() == q.time_point.interval_hull()
                        && p.time_interval.interval_hull() == q.time_interval.interval_hull()
                })
        })
}

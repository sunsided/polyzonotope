//! Mutation checks: each term of the abstraction error is needed. Dropping
//! one makes a sampled trajectory leave the sets in a scenario chosen for
//! that term, while the full error contains every sample there.
//!
//! The rounding-only terms (`JacobianSpread`, `StaticRadius`) are zero in
//! exact arithmetic and far below the sampling tolerance in `f64`, so no
//! simulation can show them; they are not checked here.

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};
use polyzonotope::{IntervalBox, SparsePolyZonotope, Zonotope};

use super::{Method, NonlinearReach, NonlinearSettings, Term, Terms};
use crate::Polynomial;

/// A vector field in plain `f64`: state and scalar input to derivative.
type Field = fn(&[f64; 2], f64) -> [f64; 2];

/// Absolute tolerance for containment, covering the RK4 reference's error.
const TOL: f64 = 1e-9;

/// RK4 sub-steps per reachability step.
const SUBSTEPS: u32 = 40;

/// Trajectories sampled per scenario.
const SAMPLES: usize = 24;

/// One test system with its start set and input range.
struct Scenario {
    /// The dynamics, two states and one input.
    dynamics: Polynomial,
    /// The same field in plain `f64`, for the reference trajectories.
    field: Field,
    /// Center of the start box.
    center: [f64; 2],
    /// Half-width of the start box.
    radius: f64,
    /// Inputs jump in `[-input, input]`.
    input: f64,
    /// Time step.
    step: f64,
    /// Number of steps.
    steps: usize,
}

impl Scenario {
    /// Whether some sampled trajectory leaves the sets computed with
    /// `method` and only the error terms in `terms`, over the steps that
    /// converged. A missing term can make a later step diverge; the steps
    /// before it still have to hold every trajectory.
    fn escapes(&self, method: Method, terms: Terms) -> bool {
        self.escapes_within(method, terms).0
    }

    /// [`escapes`](Self::escapes), and how many steps converged.
    fn escapes_within(&self, method: Method, terms: Terms) -> (bool, usize) {
        let settings = NonlinearSettings {
            taylor_terms: 12,
            max_pieces: 1,
            method,
            ..NonlinearSettings::new(self.step)
        };
        let reach = NonlinearReach::new(self.dynamics.clone(), settings);
        let u = Zonotope::new(DVector::zeros(1), DMatrix::from_element(1, 1, self.input));
        let mut x = start_box(self.center, self.radius);
        let mut sets = Vec::with_capacity(self.steps);
        for _ in 0..self.steps {
            let Ok(step) = reach.step_with(&x, &u, terms) else {
                break;
            };
            sets.push((
                Enclosure::of(&step.time_interval),
                Enclosure::of(&step.time_point),
            ));
            x = step.time_point;
        }

        let escaped = (0..SAMPLES).any(|s| self.sample_escapes(s, &sets));
        (escaped, sets.len())
    }

    /// Whether trajectory `s` leaves one of `sets`. Samples start at the
    /// corners first, then at random points. The input sits at an end of
    /// its range, where it pushes hardest: held at `+input` or `−input` for
    /// two of every three samples, jumping between the two in the third.
    fn sample_escapes(&self, s: usize, sets: &[(Enclosure, Enclosure)]) -> bool {
        let mut rng = Lcg(u64::try_from(s).expect("small") + 11);
        let mut x = [0.0; 2];
        for (i, xi) in x.iter_mut().enumerate() {
            let corner = if (s >> i) & 1 == 0 { -1.0 } else { 1.0 };
            *xi = self.center[i] + self.radius * if s < 4 { corner } else { rng.next() };
        }

        let h = self.step / f64::from(SUBSTEPS);
        for (during, end) in sets {
            for _ in 0..SUBSTEPS {
                let u = match s % 3 {
                    0 => self.input,
                    1 => -self.input,
                    _ if rng.next() < 0.0 => -self.input,
                    _ => self.input,
                };
                x = rk4(self.field, &x, u, h);
                if !during.holds(&x) {
                    return true;
                }
            }

            if !end.holds(&x) {
                return true;
            }
        }

        false
    }

    /// Asserts that the full error contains every sample and that dropping
    /// `term` lets one escape.
    fn shows(&self, method: Method, term: Term) {
        let (escaped, converged) = self.escapes_within(method, Terms::ALL);
        assert_eq!(converged, self.steps, "the full error must converge");
        assert!(!escaped, "the full error must hold");
        assert!(
            self.escapes(method, Terms::ALL.without(term)),
            "dropping {term:?} must let a trajectory escape"
        );
    }
}

/// A set's interval hull and zonotope enclosure, a necessary test for
/// membership.
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
    /// function in 32 directions.
    fn holds(&self, x: &[f64; 2]) -> bool {
        let p = DVector::from_column_slice(x);
        let z = &self.zonotope;
        self.hull.contains(&p, TOL)
            && (0..32).all(|k| {
                let angle = core::f64::consts::TAU * f64::from(k) / 32.0;
                let d = DVector::from_vec(vec![libm::cos(angle), libm::sin(angle)]);
                let support = d.dot(z.center()) + (d.transpose() * z.generators()).abs().sum();
                d.dot(&p) <= support + TOL
            })
    }
}

/// A linear congruential generator in `[-1, 1]`.
struct Lcg(u64);

impl Lcg {
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
fn rk4(f: Field, x: &[f64; 2], u: f64, h: f64) -> [f64; 2] {
    let at = |base: &[f64; 2], k: &[f64; 2], c: f64| [base[0] + c * k[0], base[1] + c * k[1]];
    let k1 = f(x, u);
    let k2 = f(&at(x, &k1, h / 2.0), u);
    let k3 = f(&at(x, &k2, h / 2.0), u);
    let k4 = f(&at(x, &k3, h), u);
    [0, 1].map(|i| x[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

/// The box `center ± radius` with dependent factors.
fn start_box(center: [f64; 2], radius: f64) -> SparsePolyZonotope {
    SparsePolyZonotope::from_zonotope(&Zonotope::new(
        DVector::from_column_slice(&center),
        DMatrix::from_diagonal_element(2, 2, radius),
    ))
}

/// Van der Pol with an additive input.
fn van_der_pol() -> Polynomial {
    Polynomial::new(2, 1)
        .term(0, 1.0, &[0, 1, 0])
        .term(1, 1.0, &[0, 1, 0])
        .term(1, -1.0, &[2, 1, 0])
        .term(1, -1.0, &[1, 0, 0])
        .term(1, 1.0, &[0, 0, 1])
}

/// [`van_der_pol`] in plain `f64`.
fn van_der_pol_f64(x: &[f64; 2], u: f64) -> [f64; 2] {
    [x[1], (1.0 - x[0] * x[0]) * x[1] - x[0] + u]
}

/// Van der Pol from a wide box.
fn wide_van_der_pol() -> Scenario {
    Scenario {
        dynamics: van_der_pol(),
        field: van_der_pol_f64,
        center: [1.4, 2.4],
        radius: 0.2,
        input: 0.0,
        step: 0.02,
        steps: 20,
    }
}

/// Van der Pol from a single point.
fn point_van_der_pol() -> Scenario {
    Scenario {
        radius: 0.0,
        step: 0.05,
        ..wide_van_der_pol()
    }
}

/// `x₀' = x₀³`, `x₁' = −x₁` around the origin, where the Hessian vanishes.
fn cubic() -> Scenario {
    Scenario {
        dynamics: Polynomial::new(2, 1)
            .term(0, 1.0, &[3, 0, 0])
            .term(1, -1.0, &[0, 1, 0]),
        field: |x, _| [x[0] * x[0] * x[0], -x[1]],
        center: [0.0, 0.0],
        radius: 0.5,
        input: 0.0,
        step: 0.05,
        steps: 10,
    }
}

/// `x₀' = x₀ x₁`, `x₁' = 1 + x₀` from a box around the origin: the error
/// `(x₀ − x₀*)(x₁ − x₁*)` is all cross term at first, since `x₁` starts as
/// a point and races ahead fastest where `x₀` is largest, so `δ` does not
/// average out over the step.
fn shear() -> Scenario {
    Scenario {
        dynamics: Polynomial::new(2, 1)
            .term(0, 1.0, &[1, 1, 0])
            .term(1, 1.0, &[0, 0, 0])
            .term(1, 1.0, &[1, 0, 0]),
        field: |x, _| [x[0] * x[1], 1.0 + x[0]],
        center: [0.0, 0.0],
        radius: 0.2,
        input: 0.0,
        step: 0.05,
        steps: 20,
    }
}

/// `x₀' = 0`, `x₁' = x₀ u`: the input gain is the state. The linear model
/// sees the gain `x₀*` only; the rest, `(x₀ − x₀*) u`, comes from `s` times
/// the input part of `δ`.
fn modulated() -> Scenario {
    Scenario {
        dynamics: Polynomial::new(2, 1).term(1, 1.0, &[1, 0, 1]),
        field: |x, u| [0.0, x[0] * u],
        center: [1.0, 0.0],
        radius: 0.5,
        input: 0.5,
        step: 0.05,
        steps: 20,
    }
}

/// Linearization: the quadratic error carries the curvature of a wide set.
#[test]
fn linearization_needs_the_quadratic_error() {
    wide_van_der_pol().shows(Method::Linearization, Term::Quadratic);
}

/// Linearization: at the origin the cubic is invisible to the Hessian's
/// center, only its spread holds the drift. A narrower box than for
/// polynomialization: from radius 0.5, linearization needs splitting.
#[test]
fn linearization_needs_the_hessian_spread() {
    Scenario {
        radius: 0.3,
        ..cubic()
    }
    .shows(Method::Linearization, Term::HessianSpread);
}

/// Polynomialization: the static polynomial carries the curvature of a
/// wide set.
#[test]
fn polynomialization_needs_the_static_set() {
    wide_van_der_pol().shows(Method::Polynomialization, Term::StaticSet);
}

/// Polynomialization: the cross term couples the width of the set with its
/// motion during the step.
#[test]
fn polynomialization_needs_the_cross_term() {
    shear().shows(Method::Polynomialization, Term::Cross);
}

/// Polynomialization: from a point, the motion during the step is all
/// there is.
#[test]
fn polynomialization_needs_the_dynamic_quadratic() {
    point_van_der_pol().shows(Method::Polynomialization, Term::DynamicQuadratic);
}

/// Polynomialization: at the origin the cubic has no second-order part, so
/// only the third-order remainder holds the drift.
#[test]
fn polynomialization_needs_the_remainder() {
    cubic().shows(Method::Polynomialization, Term::Remainder);
}

/// Polynomialization: the input varying during the step enters through
/// `∂²f/∂x∂u`.
#[test]
fn polynomialization_needs_the_input_deviation() {
    modulated().shows(Method::Polynomialization, Term::InputDeviation);
}

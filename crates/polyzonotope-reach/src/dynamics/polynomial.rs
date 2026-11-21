//! Polynomial vector fields, with exact symbolic derivatives.

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::DMatrix;
use polyzonotope::{Interval, IntervalMatrix};

use super::{Dynamics, IntervalTensor};

/// Polynomial dynamics: each `f_i(z)` is a sum of terms `c · Π_j z_j^{e_j}`
/// over `z = (x, u)`.
///
/// The Jacobian, the Hessians and the third derivatives are derived term by term, so only `f` has to be
/// written down. Every evaluation runs in interval arithmetic, outward
/// rounded with the `sound` feature, and even powers of an interval that
/// spans zero start at zero.
///
/// # Examples
///
/// Van der Pol, `x₀' = x₁`, `x₁' = μ (1 − x₀²) x₁ − x₀`, with `μ = 1`:
///
/// ```
/// use polyzonotope_reach::{Dynamics, Polynomial};
/// use polyzonotope_reach::polyzonotope::Interval;
///
/// let vdp = Polynomial::new(2, 0)
///     .term(0, 1.0, &[0, 1])
///     .term(1, 1.0, &[0, 1])
///     .term(1, -1.0, &[2, 1])
///     .term(1, -1.0, &[1, 0]);
///
/// let z = [Interval::point(2.0), Interval::point(1.0)];
/// assert_eq!(vdp.flow(&z)[1], Interval::point(-5.0));
/// assert_eq!(vdp.jacobian(&z).0[(1, 0)], Interval::point(-5.0));
/// assert_eq!(vdp.hessians(&z)[1].0[(0, 0)], Interval::point(-2.0));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Polynomial {
    /// Number of states `n`.
    states: usize,
    /// Number of inputs `m`.
    inputs: usize,
    /// The terms of `f_i` in row `i`; there are `n` rows.
    rows: Vec<Vec<Term>>,
}

/// The whole real line, the enclosure of a value that overflowed.
const UNBOUNDED: Interval = Interval {
    lo: f64::NEG_INFINITY,
    hi: f64::INFINITY,
};

/// One term `c · Π_j z_j^{e_j}`.
#[derive(Clone, Debug, PartialEq)]
struct Term {
    /// The coefficient `c`, as an interval so that derivatives can scale it
    /// without rounding inward.
    coefficient: Interval,
    /// The exponents `e_j`, one per entry of `z = (x, u)`.
    exponents: Vec<u32>,
}

impl Polynomial {
    /// The zero vector field with `states` states and `inputs` inputs; add
    /// terms with [`term`](Self::term).
    #[must_use]
    pub fn new(states: usize, inputs: usize) -> Self {
        Self {
            states,
            inputs,
            rows: vec![Vec::new(); states],
        }
    }

    /// The zero vector field of an autonomous system `x' = f(x)` with
    /// `states` states; the same as `new(states, 0)`.
    #[must_use]
    pub fn autonomous(states: usize) -> Self {
        Self::new(states, 0)
    }

    /// Adds `coefficient · Π_j z_j^{exponents[j]}` to `f_row`, where
    /// `z = (x, u)`.
    ///
    /// # Panics
    ///
    /// Panics if `row` is not a state, if `exponents` does not have one
    /// entry per state and input, or if `coefficient` is not finite.
    #[must_use]
    pub fn term(mut self, row: usize, coefficient: f64, exponents: &[u32]) -> Self {
        assert!(row < self.states, "row {row} is not a state");
        assert_eq!(
            exponents.len(),
            self.states + self.inputs,
            "one exponent per state and input"
        );
        assert!(coefficient.is_finite(), "coefficient must be finite");
        self.rows[row].push(Term {
            coefficient: Interval::point(coefficient),
            exponents: exponents.to_vec(),
        });

        self
    }

    /// Number of entries of `z = (x, u)`.
    fn width(&self) -> usize {
        self.states + self.inputs
    }

    /// The sum of the terms of `f_row`, each first differentiated by the
    /// variables in `by`; the whole real line once a bound overflows, since
    /// under `sound` adding an infinite bound would panic.
    fn evaluate(&self, row: usize, by: &[usize], z: &[Interval]) -> Interval {
        let finite = |iv: Interval| iv.lo.is_finite() && iv.hi.is_finite();
        let mut sum = Interval::point(0.0);
        for term in self.rows[row].iter().filter_map(|t| t.derivative(by)) {
            let value = term.value(z);
            if !finite(value) {
                return UNBOUNDED;
            }

            sum = sum + value;
            if !finite(sum) {
                return UNBOUNDED;
            }
        }

        sum
    }
}

impl Polynomial {
    /// `∂³f_row/∂z³` over the box `z`. Only the entries with `a ≤ b ≤ c`
    /// are evaluated; the others are copies, since mixed partial derivatives
    /// of a polynomial commute.
    fn third_derivative(&self, row: usize, z: &[Interval]) -> IntervalTensor {
        let w = self.width();
        let mut t = IntervalTensor::zeros(w);
        for a in 0..w {
            for b in a..w {
                for c in b..w {
                    let value = self.evaluate(row, &[a, b, c], z);
                    for at in permutations(a, b, c) {
                        t[at] = value;
                    }
                }
            }
        }

        t
    }
}

/// The six orderings of `(a, b, c)`, repeats included.
fn permutations(a: usize, b: usize, c: usize) -> [(usize, usize, usize); 6] {
    [
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a),
    ]
}

impl Dynamics for Polynomial {
    fn states(&self) -> usize {
        self.states
    }

    fn inputs(&self) -> usize {
        self.inputs
    }

    fn flow(&self, z: &[Interval]) -> Vec<Interval> {
        (0..self.states).map(|i| self.evaluate(i, &[], z)).collect()
    }

    fn jacobian(&self, z: &[Interval]) -> IntervalMatrix {
        IntervalMatrix(DMatrix::from_fn(self.states, self.width(), |i, j| {
            self.evaluate(i, &[j], z)
        }))
    }

    fn hessians(&self, z: &[Interval]) -> Vec<IntervalMatrix> {
        let w = self.width();
        (0..self.states)
            .map(|i| IntervalMatrix(DMatrix::from_fn(w, w, |a, b| self.evaluate(i, &[a, b], z))))
            .collect()
    }

    fn third_derivatives(&self, z: &[Interval]) -> Option<Vec<IntervalTensor>> {
        Some(
            (0..self.states)
                .map(|i| self.third_derivative(i, z))
                .collect(),
        )
    }
}

impl Term {
    /// The term differentiated once by each variable in `by`, or `None` if
    /// that makes it zero.
    fn derivative(&self, by: &[usize]) -> Option<Self> {
        let mut d = self.clone();
        for &j in by {
            let e = d.exponents[j];
            if e == 0 {
                return None;
            }

            d.coefficient = d.coefficient * Interval::point(f64::from(e));
            d.exponents[j] = e - 1;
        }

        Some(d)
    }

    /// `c · Π_j z_j^{e_j}` over the box `z`.
    fn value(&self, z: &[Interval]) -> Interval {
        self.exponents
            .iter()
            .zip(z)
            .filter(|&(&e, _)| e > 0)
            .fold(self.coefficient, |acc, (&e, &x)| acc * power(x, e))
    }
}

/// `{ x^k | x ∈ x }`: tight for monotone odd powers and for even powers,
/// which are non-negative.
fn power(x: Interval, k: u32) -> Interval {
    let up = |v: f64| point_power(v, k).hi;
    let down = |v: f64| point_power(v, k).lo;
    if k == 0 {
        Interval::point(1.0)
    } else if k % 2 == 1 || x.lo >= 0.0 {
        Interval::new(down(x.lo), up(x.hi))
    } else if x.hi <= 0.0 {
        Interval::new(down(-x.hi), up(-x.lo))
    } else {
        Interval::new(0.0, up(x.mag()))
    }
}

/// `v^k` by repeated interval multiplication, so that rounding is covered.
fn point_power(v: f64, k: u32) -> Interval {
    (0..k).fold(Interval::point(1.0), |acc, _| acc * Interval::point(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Even powers of an interval across zero start at zero; odd ones keep
    /// their sign.
    #[test]
    fn powers_respect_parity() {
        let x = Interval::new(-2.0, 1.0);
        assert_eq!(power(x, 2), Interval::new(0.0, 4.0));
        assert_eq!(power(x, 3), Interval::new(-8.0, 1.0));
        assert_eq!(power(Interval::new(-3.0, -1.0), 2), Interval::new(1.0, 9.0));
        assert_eq!(power(x, 0), Interval::point(1.0));
    }

    /// The derivatives of `x₀² x₁ + 3 x₁` match the hand-computed ones at a
    /// point, and the Hessian is symmetric.
    #[test]
    fn derivatives_match_by_hand() {
        let p = Polynomial::new(1, 1)
            .term(0, 1.0, &[2, 1])
            .term(0, 3.0, &[0, 1]);
        let z = [Interval::point(2.0), Interval::point(5.0)];

        assert_eq!(p.flow(&z)[0], Interval::point(35.0));
        let j = p.jacobian(&z);
        assert_eq!(j.0[(0, 0)], Interval::point(20.0));
        assert_eq!(j.0[(0, 1)], Interval::point(7.0));
        let h = &p.hessians(&z)[0];
        assert_eq!(h.0[(0, 0)], Interval::point(10.0));
        assert_eq!(h.0[(0, 1)], Interval::point(4.0));
        assert_eq!(h.0[(1, 0)], Interval::point(4.0));
        assert_eq!(h.0[(1, 1)], Interval::point(0.0));
    }

    /// The third derivatives of `x₀² x₁ + x₀³ x₁²` match the hand-computed
    /// ones at a point, in every ordering of the indices.
    #[test]
    fn third_derivatives_match_by_hand() {
        let p = Polynomial::new(1, 1)
            .term(0, 1.0, &[2, 1])
            .term(0, 1.0, &[3, 2]);
        let z = [Interval::point(2.0), Interval::point(3.0)];
        let t = &p.third_derivatives(&z).expect("polynomials supply them")[0];

        // ∂³/∂x₀³ = 6 x₁², ∂³/∂x₀²∂x₁ = 2 + 12 x₀ x₁, ∂³/∂x₀∂x₁² = 6 x₀², ∂³/∂x₁³ = 0.
        assert_eq!(t[(0, 0, 0)], Interval::point(54.0));
        for at in [(0, 0, 1), (0, 1, 0), (1, 0, 0)] {
            assert_eq!(t[at], Interval::point(74.0));
        }
        for at in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert_eq!(t[at], Interval::point(24.0));
        }
        assert_eq!(t[(1, 1, 1)], Interval::point(0.0));
    }

    /// A value too large for `f64` gives the whole real line, not a panic.
    #[test]
    fn overflow_is_unbounded() {
        let field = Polynomial::new(1, 0).term(0, 1.0, &[9]).term(0, 1.0, &[1]);
        let flow = field.flow(&[Interval::new(-1e40, 1e40)]);
        assert_eq!(flow[0], UNBOUNDED);
    }

    /// The enclosure over a box contains the value at sampled points.
    #[test]
    fn box_enclosure_contains_samples() {
        let field = Polynomial::new(2, 0)
            .term(0, -1.5, &[3, 1])
            .term(1, 0.5, &[1, 2]);
        let flow = field.flow(&[Interval::new(-1.0, 0.5), Interval::new(0.2, 2.0)]);
        for i in 0..=10 {
            for j in 0..=10 {
                let x = -1.0 + 0.15 * f64::from(i);
                let y = 0.2 + 0.18 * f64::from(j);
                assert!(flow[0].contains(-1.5 * x * x * x * y, 1e-12));
                assert!(flow[1].contains(0.5 * x * y * y, 1e-12));
            }
        }
    }
}

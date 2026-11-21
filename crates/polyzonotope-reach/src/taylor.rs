//! Truncated Taylor series of the matrix exponential, with the remainder as
//! an interval matrix (Althoff 2010, Sec. 3.2).

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::DMatrix;
use polyzonotope::{Interval, IntervalMatrix};

use crate::ReachError;

/// The series `Σ_{i=0}^{η} (A r)^i / i!` term by term, with the bound `E(r)`
/// on everything after the last term.
pub(crate) struct Taylor {
    /// `A^i` for `i = 0..=η+1`, in interval arithmetic.
    pub(crate) powers: Vec<IntervalMatrix>,
    /// Time step `r`.
    pub(crate) step: f64,
    /// `E(r) = [-φ, φ]` entrywise, `φ = (‖A‖∞ r)^{η+1} / (η+1)! · 1 / (1 − ε)`,
    /// `ε = ‖A‖∞ r / (η + 2)`.
    pub(crate) remainder: IntervalMatrix,
}

impl Taylor {
    /// The series for `A` over the step `r` with `terms = η` terms after the first.
    ///
    /// # Errors
    ///
    /// [`ReachError::InvalidStep`] unless `r` is finite and non-negative: a
    /// negative `r` makes `ε` negative and the remainder bound too small.
    /// [`ReachError::StepTooLarge`] if `ε >= 1`, where the remainder bound
    /// does not hold; use a smaller step or more terms.
    pub(crate) fn new(a: &DMatrix<f64>, step: f64, terms: usize) -> Result<Self, ReachError> {
        if !(step.is_finite() && step >= 0.0) {
            return Err(ReachError::InvalidStep);
        }
        let n = a.nrows();
        let mut powers = vec![IntervalMatrix::point(&DMatrix::identity(n, n))];
        let a_iv = IntervalMatrix::point(a);
        for _ in 0..=terms {
            let next = powers.last().expect("starts with the identity").mul(&a_iv);
            powers.push(next);
        }

        let norm_r = infinity_norm(a) * Interval::point(step);
        let eps = norm_r / Interval::point(count(terms + 2));
        if eps.hi >= 1.0 {
            return Err(ReachError::StepTooLarge);
        }

        let tail = pow(norm_r, terms + 1) / factorial(terms + 1) / (Interval::point(1.0) - eps);
        let phi = tail.hi;
        let remainder = IntervalMatrix(DMatrix::from_element(n, n, Interval::new(-phi, phi)));

        Ok(Self {
            powers,
            step,
            remainder,
        })
    }

    /// Number of terms `η` after the first.
    pub(crate) fn terms(&self) -> usize {
        self.powers.len() - 2
    }

    /// `e^{A r} ⊆ Σ_{i=0}^{η} A^i r^i / i! + E(r)`.
    pub(crate) fn exponential(&self) -> IntervalMatrix {
        (0..=self.terms())
            .map(|i| self.powers[i].scale(pow(self.r(), i) / factorial(i)))
            .fold(self.remainder.clone(), |acc, t| acc.add(&t))
    }

    /// `Γ(r) = ∫_0^r e^{A s} ds ⊆ Σ_{i=0}^{η} A^i r^{i+1} / (i+1)! + E(r) r`,
    /// which maps a constant input to its effect after one step.
    pub(crate) fn input_solution(&self) -> IntervalMatrix {
        (0..=self.terms())
            .map(|i| self.input_term(i))
            .fold(self.remainder.scale(self.r()), |acc, t| acc.add(&t))
    }

    /// `A^i r^{i+1} / (i+1)!`, the `i`-th term of [`input_solution`](Self::input_solution).
    pub(crate) fn input_term(&self, i: usize) -> IntervalMatrix {
        self.powers[i].scale(pow(self.r(), i + 1) / factorial(i + 1))
    }

    /// `F = Σ_{i=2}^{η} [(i^{-i/(i-1)} − i^{-1/(i-1)}) r^i, 0] A^i / i! + E(r)`:
    /// for `t ∈ [0, r]`, `e^{A t} x` lies in the convex hull of `x` and `e^{A r} x`
    /// plus `F x`.
    pub(crate) fn state_correction(&self) -> IntervalMatrix {
        (2..=self.terms())
            .map(|i| self.powers[i].scale(correction_factor(i) * pow(self.r(), i) / factorial(i)))
            .fold(self.remainder.clone(), |acc, t| acc.add(&t))
    }

    /// `F̃ = Σ_{i=2}^{η+1} [(i^{-i/(i-1)} − i^{-1/(i-1)}) r^i, 0] A^{i-1} / i! + E(r) r`:
    /// the same correction for the response to a constant input.
    pub(crate) fn input_correction(&self) -> IntervalMatrix {
        (2..=self.terms() + 1)
            .map(|i| {
                self.powers[i - 1].scale(correction_factor(i) * pow(self.r(), i) / factorial(i))
            })
            .fold(self.remainder.scale(self.r()), |acc, t| acc.add(&t))
    }

    /// The step `r` as an interval.
    fn r(&self) -> Interval {
        Interval::point(self.step)
    }
}

/// `[i^{-i/(i-1)} − i^{-1/(i-1)}, 0]`, widened outward by a relative `1e-12`
/// because `powf` is not correctly rounded.
fn correction_factor(i: usize) -> Interval {
    let x = count(i);
    let e = 1.0 / (x - 1.0);
    let lo = libm::pow(x, -x * e) - libm::pow(x, -e);
    // shortcut: libm's pow is accurate to an ulp or two; 1e-12 covers it with room to spare.
    Interval::new(lo * (1.0 + 1e-12), 0.0)
}

/// `‖A‖∞ = max_i Σ_j |A_ij|`, rounded up.
fn infinity_norm(a: &DMatrix<f64>) -> Interval {
    let rows = a.row_iter().map(|row| {
        row.iter()
            .fold(Interval::point(0.0), |acc, x| {
                acc + Interval::point(x.abs())
            })
            .hi
    });
    Interval::point(rows.fold(0.0, f64::max))
}

/// `x^k` by repeated interval multiplication.
fn pow(x: Interval, k: usize) -> Interval {
    (0..k).fold(Interval::point(1.0), |acc, _| acc * x)
}

/// `k!` as an interval.
fn factorial(k: usize) -> Interval {
    (1..=k).fold(Interval::point(1.0), |acc, j| {
        acc * Interval::point(count(j))
    })
}

/// A small count as `f64`.
#[expect(
    clippy::cast_precision_loss,
    reason = "term counts stay far below 2^52"
)]
fn count(k: usize) -> f64 {
    k as f64
}

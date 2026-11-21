//! Nonlinear dynamics `x' = f(x, u)` with their first, second and
//! (optionally) third derivatives, all enclosed in interval arithmetic.

use alloc::vec::Vec;

use polyzonotope::{Interval, IntervalMatrix};

mod polynomial;
mod tensor;

pub use polynomial::Polynomial;
pub use tensor::IntervalTensor;

/// A vector field `x' = f(x, u)` that can be enclosed over boxes.
///
/// Every method takes `z = (x, u)`: first the `states()` state intervals,
/// then the `inputs()` input intervals. Each result must contain the exact
/// value for every point of the box `z`. A point interval asks for the value
/// at that point; the enclosure then only has to cover rounding. When a
/// bound overflows, return an infinite one instead of panicking: the step
/// then refuses with [`ReachError::AbstractionDiverged`](crate::ReachError::AbstractionDiverged).
/// Under `sound`, the interval operations of `polyzonotope` panic when an
/// operand is infinite, so check before adding to an overflowed value.
///
/// [`Polynomial`] implements this for polynomial dynamics. For other
/// functions, implement it with interval extensions of the elementary
/// functions; a loose enclosure stays sound, it only makes the sets larger.
///
/// # Examples
///
/// The undamped Duffing oscillator `x₀' = x₁, x₁' = −x₀³`, by hand:
///
/// ```
/// use polyzonotope_reach::Dynamics;
/// use polyzonotope_reach::polyzonotope::{Interval, IntervalMatrix};
///
/// struct Duffing;
///
/// impl Dynamics for Duffing {
///     fn states(&self) -> usize { 2 }
///     fn inputs(&self) -> usize { 0 }
///     fn flow(&self, z: &[Interval]) -> Vec<Interval> {
///         vec![z[1], -(z[0] * z[0] * z[0])]
///     }
///     fn jacobian(&self, z: &[Interval]) -> IntervalMatrix {
///         let mut j = IntervalMatrix::zeros(2, 2);
///         j.0[(0, 1)] = Interval::point(1.0);
///         j.0[(1, 0)] = Interval::point(-3.0) * z[0] * z[0];
///         j
///     }
///     fn hessians(&self, z: &[Interval]) -> Vec<IntervalMatrix> {
///         let mut h = IntervalMatrix::zeros(2, 2);
///         h.0[(0, 0)] = Interval::point(-6.0) * z[0];
///         vec![IntervalMatrix::zeros(2, 2), h]
///     }
/// }
///
/// let f = Duffing.flow(&[Interval::point(2.0), Interval::point(1.0)]);
/// assert_eq!(f[1], Interval::point(-8.0));
/// ```
pub trait Dynamics {
    /// Number of states `n`.
    fn states(&self) -> usize;

    /// Number of inputs `m`; zero for an autonomous system.
    fn inputs(&self) -> usize;

    /// `f(z)`, one interval per state.
    fn flow(&self, z: &[Interval]) -> Vec<Interval>;

    /// The Jacobian `∂f/∂z`, `n × (n + m)`: the state columns first, then the
    /// input columns.
    fn jacobian(&self, z: &[Interval]) -> IntervalMatrix;

    /// The Hessians `∂²f_i/∂z²`, one `(n + m) × (n + m)` matrix per state `i`.
    fn hessians(&self, z: &[Interval]) -> Vec<IntervalMatrix>;

    /// The third derivatives `∂³f_i/∂z³`, one `(n + m)³` tensor per state
    /// `i`, or `None` if the dynamics do not supply them.
    ///
    /// Only [`Method::Polynomialization`](crate::Method::Polynomialization)
    /// needs them, for its third-order Lagrange remainder; without them it
    /// refuses with
    /// [`ReachError::MissingDerivatives`](crate::ReachError::MissingDerivatives).
    /// The default returns `None`, so existing implementations keep working
    /// with [`Method::Linearization`](crate::Method::Linearization).
    fn third_derivatives(&self, z: &[Interval]) -> Option<Vec<IntervalTensor>> {
        let _ = z;
        None
    }
}

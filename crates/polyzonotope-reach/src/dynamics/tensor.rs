//! Cubic arrays of intervals, for third derivatives.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use polyzonotope::Interval;

/// A `d × d × d` array of intervals: the set of all real arrays with each
/// entry in its interval.
///
/// [`Dynamics::third_derivatives`](crate::Dynamics::third_derivatives)
/// returns one per state `i`, with entry `(a, b, c)` enclosing
/// `∂³f_i / ∂z_a ∂z_b ∂z_c`. For a three times continuously differentiable
/// `f` the true array is symmetric, but nothing here relies on that: the
/// remainder sums every entry.
///
/// # Examples
///
/// ```
/// use polyzonotope_reach::IntervalTensor;
/// use polyzonotope_reach::polyzonotope::Interval;
///
/// let mut t = IntervalTensor::zeros(2);
/// t[(0, 0, 1)] = Interval::point(-2.0);
/// assert_eq!(t[(0, 0, 1)], Interval::point(-2.0));
/// assert_eq!(t[(1, 0, 0)], Interval::point(0.0));
/// assert_eq!(t.dim(), 2);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalTensor {
    /// The side length `d`.
    dim: usize,
    /// The `d³` entries, `(a, b, c)` at `(a · d + b) · d + c`.
    entries: Vec<Interval>,
}

impl IntervalTensor {
    /// The `dim × dim × dim` array of zeros.
    #[must_use]
    pub fn zeros(dim: usize) -> Self {
        Self {
            dim,
            entries: vec![Interval::point(0.0); dim * dim * dim],
        }
    }

    /// The side length `d`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The position of entry `(a, b, c)` in `entries`.
    ///
    /// # Panics
    ///
    /// Panics if an index is not below [`dim`](Self::dim).
    fn offset(&self, (a, b, c): (usize, usize, usize)) -> usize {
        let d = self.dim;
        assert!(
            a < d && b < d && c < d,
            "index ({a}, {b}, {c}) out of a {d}³ tensor"
        );
        (a * d + b) * d + c
    }
}

impl Index<(usize, usize, usize)> for IntervalTensor {
    type Output = Interval;

    /// Entry `(a, b, c)`; panics if an index is not below the side length.
    fn index(&self, at: (usize, usize, usize)) -> &Interval {
        &self.entries[self.offset(at)]
    }
}

impl IndexMut<(usize, usize, usize)> for IntervalTensor {
    /// Entry `(a, b, c)`; panics if an index is not below the side length.
    fn index_mut(&mut self, at: (usize, usize, usize)) -> &mut Interval {
        let offset = self.offset(at);
        &mut self.entries[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry has its own slot.
    #[test]
    fn entries_are_distinct() {
        let mut t = IntervalTensor::zeros(3);
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    t[(a, b, c)] =
                        Interval::point(f64::from(u8::try_from(9 * a + 3 * b + c).unwrap()));
                }
            }
        }

        assert_eq!(t[(2, 1, 0)], Interval::point(21.0));
        assert_eq!(t[(0, 1, 2)], Interval::point(5.0));
    }

    /// An index past the side length panics instead of aliasing another entry.
    #[test]
    #[should_panic(expected = "out of a 2³ tensor")]
    fn out_of_range_panics() {
        let _ = IntervalTensor::zeros(2)[(0, 2, 0)];
    }
}

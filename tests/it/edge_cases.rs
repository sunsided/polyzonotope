//! Edge cases of single operations: degenerate inputs and the panics they document.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{Interval, Zonotope};

/// The order is the number of generators per dimension.
#[test]
fn order_is_generators_per_dimension() {
    let z = Zonotope::new(DVector::zeros(2), DMatrix::zeros(2, 3));

    assert!((z.order() - 1.5).abs() < f64::EPSILON);
}

/// A zero-dimensional zonotope has no order, and says so instead of returning NaN.
#[test]
#[should_panic(expected = "order of a zero-dimensional zonotope")]
fn order_of_a_zero_dimensional_zonotope_panics() {
    let _ = Zonotope::new(DVector::zeros(0), DMatrix::zeros(0, 0)).order();
}

/// A NaN point is rejected like a NaN bound in [`Interval::new`].
#[test]
#[should_panic(expected = "interval bound is NaN")]
fn nan_point_panics() {
    let _ = Interval::point(f64::NAN);
}

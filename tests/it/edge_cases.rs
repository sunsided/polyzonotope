//! Edge cases of single operations: degenerate inputs and the panics they document.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, Interval, ShapeError, SparsePolyZonotope, Zonotope};

/// The parts of a 2-D set before [`SparsePolyZonotope::new`] checks them.
struct Parts {
    /// Dependent generators, `2 × h`.
    g: DMatrix<f64>,
    /// Independent generators, `2 × q`.
    gi: DMatrix<f64>,
    /// Exponents, one row per id and one column per dependent generator.
    e: DMatrix<u32>,
    /// Factor ids, one per row of `e`.
    id: Vec<FactorId>,
}

impl Parts {
    /// One dependent and one independent generator, with consistent shapes.
    fn consistent() -> Self {
        Self {
            g: DMatrix::from_element(2, 1, 1.0),
            gi: DMatrix::from_element(2, 1, 0.5),
            e: DMatrix::from_element(1, 1, 1),
            id: FactorId::fresh_n(1),
        }
    }

    /// Hands the parts to [`SparsePolyZonotope::new`] with a 2-D zero center.
    fn build(self) -> Result<SparsePolyZonotope, ShapeError> {
        SparsePolyZonotope::new(DVector::zeros(2), self.g, self.gi, self.e, self.id)
    }
}

/// A change to [`Parts`] that breaks exactly one shape check.
type BreakShape = fn(&mut Parts);

/// Every shape check of [`SparsePolyZonotope::new`] reports its own error,
/// with a message, and consistent parts pass.
#[test]
fn new_reports_each_shape_error() {
    let cases: [(ShapeError, BreakShape); 5] = [
        (ShapeError::GeneratorRows, |p| {
            p.g = DMatrix::zeros(3, 1);
        }),
        (ShapeError::IndependentRows, |p| {
            p.gi = DMatrix::zeros(3, 1);
        }),
        (ShapeError::ExponentColumns, |p| {
            p.e = DMatrix::zeros(1, 2);
        }),
        (ShapeError::ExponentRows, |p| {
            p.id.push(FactorId::fresh());
        }),
        (ShapeError::DuplicateId, |p| {
            p.e = DMatrix::zeros(2, 1);
            p.id.push(p.id[0]);
        }),
    ];

    assert!(Parts::consistent().build().is_ok());
    for (expected, break_shape) in cases {
        let mut parts = Parts::consistent();
        break_shape(&mut parts);
        let err = parts.build().expect_err("the shape is broken");

        assert_eq!(err, expected);
        assert_ne!(err.to_string(), "");
    }
}

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

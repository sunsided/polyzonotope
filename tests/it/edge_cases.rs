//! Edge cases of single operations: degenerate inputs and the panics they document.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{
    Containment, FactorId, Halfspace, Intersection, Interval, Polytope, ShapeError,
    SparsePolyZonotope, Zonotope,
};

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

/// The 1-D set `α^d`, for splitting at the exponent limit.
fn power(d: u32) -> SparsePolyZonotope {
    SparsePolyZonotope::new(
        DVector::zeros(1),
        DMatrix::from_element(1, 1, 1.0),
        DMatrix::zeros(1, 0),
        DMatrix::from_element(1, 1, d),
        FactorId::fresh_n(1),
    )
    .expect("consistent shapes")
}

/// Exponent 56 is the largest whose binomial coefficients are exact; both halves
/// of `α^56` evaluate to `((α' ∓ 1) / 2)^56`.
#[test]
fn split_accepts_the_largest_exact_exponent() {
    let pz = power(56);
    let (lower, upper) = pz.split(pz.ids()[0]);
    for (half, sign) in [(&lower, -1.0f64), (&upper, 1.0)] {
        let beta = vec![0.0; half.independent_generators().ncols()];
        let got = half.evaluate(&[0.5], &beta)[0];
        let want = f64::midpoint(0.5, sign).powi(56);
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(f64::MIN_POSITIVE),
            "{got} vs {want}"
        );
    }
}

/// Beyond exponent 56 the rescaling would round, so `split` refuses.
#[test]
#[should_panic(expected = "split needs exponents of at most 56")]
fn split_rejects_inexact_exponents() {
    let pz = power(57);
    let _ = pz.split(pz.ids()[0]);
}

/// `α − α²` ranges over `[-2, 1/4]`; the plain hull says `[-2, 1]`, and
/// splitting closes the gap at the top to the tolerance.
#[test]
fn split_hull_finds_the_true_maximum() {
    let pz = alpha_minus_alpha_squared();

    let coarse = pz.interval_hull().0[0];
    let tight = pz.interval_hull_split(1e-4, 200).0[0];

    assert!((coarse.hi - 1.0).abs() < 1e-12);
    assert!(
        tight.hi >= 0.25 && tight.hi <= 0.25 + 1e-3,
        "upper bound {}",
        tight.hi
    );
    assert!((tight.lo + 2.0).abs() < 1e-3, "lower bound {}", tight.lo);
}

/// `{ α − α² | α ∈ [-1, 1] }`, which is `[-2, 1/4]`, while its plain hull is `[-2, 1]`.
fn alpha_minus_alpha_squared() -> SparsePolyZonotope {
    SparsePolyZonotope::new(
        DVector::zeros(1),
        DMatrix::from_row_slice(1, 2, &[1.0, -1.0]),
        DMatrix::zeros(1, 0),
        DMatrix::from_row_slice(1, 2, &[1, 2]),
        FactorId::fresh_n(1),
    )
    .expect("consistent shapes")
}

/// The 1-D halfspace `{ x | s · x <= d }`.
fn halfspace_1d(s: f64, d: f64) -> Polytope {
    Halfspace {
        normal: DVector::from_element(1, s),
        offset: d,
    }
    .into()
}

/// `α − α²` never reaches `x >= 0.5`: the plain hull cannot tell, splitting proves it.
#[test]
fn splitting_proves_disjointness() {
    let pz = alpha_minus_alpha_squared();
    let above_half = halfspace_1d(-1.0, -0.5);

    assert_eq!(pz.intersection(&above_half, 0), Intersection::Unknown);
    assert_eq!(pz.intersection(&above_half, 50), Intersection::Disjoint);
}

/// `α − α²` reaches `x >= 0.2` (its maximum is 0.25 at `α = 1/2`).
#[test]
fn splitting_finds_an_intersection() {
    let pz = alpha_minus_alpha_squared();

    assert_eq!(
        pz.intersection(&halfspace_1d(-1.0, -0.2), 50),
        Intersection::Intersects
    );
}

/// `α − α²` stays below 0.3 but not below 0.2; the plain hull decides neither.
#[test]
fn splitting_decides_containment() {
    let pz = alpha_minus_alpha_squared();
    let (below_03, below_02) = (halfspace_1d(1.0, 0.3), halfspace_1d(1.0, 0.2));

    assert_eq!(pz.containment(&below_03, 0), Containment::Unknown);
    assert_eq!(pz.containment(&below_03, 50), Containment::Inside);
    assert_eq!(pz.containment(&below_02, 50), Containment::NotInside);
}

/// A unit square centered at the origin misses a box beyond one facet at
/// once. For a box over its corner no witness exists without splitting, so it
/// takes a few splits to find one.
#[test]
fn zonotope_box_intersection() {
    let z = Zonotope::new(DVector::zeros(2), DMatrix::identity(2, 2));
    let overlapping =
        polyzonotope::IntervalBox(vec![Interval::new(0.5, 2.0), Interval::new(0.5, 2.0)]);
    let beyond = polyzonotope::IntervalBox(vec![Interval::new(1.5, 2.0), Interval::new(-0.5, 0.5)]);

    let overlapping = Polytope::from_box(&overlapping);
    assert_eq!(z.intersection(&overlapping, 0), Intersection::Unknown);
    assert_eq!(z.intersection(&overlapping, 8), Intersection::Intersects);
    assert_eq!(
        z.intersection(&Polytope::from_box(&beyond), 0),
        Intersection::Disjoint
    );
}

/// The unit square's polygon is its four corners, counter-clockwise from the bottom left.
#[test]
fn polygon_of_the_unit_square() {
    let z = Zonotope::new(DVector::zeros(2), DMatrix::identity(2, 2));

    assert_eq!(
        z.polygon(),
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    );
}

/// A zonotope without generators is drawn as its center.
#[test]
fn polygon_of_a_point() {
    let z = Zonotope::new(DVector::from_vec(vec![1.0, 2.0]), DMatrix::zeros(2, 0));

    assert_eq!(z.polygon(), vec![[1.0, 2.0]]);
}

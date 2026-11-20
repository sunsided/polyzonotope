//! Zonotopes: affine images of a hypercube, closed under linear maps and Minkowski sums.

use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};

use crate::reduce::{self, Reduction};
use crate::round::{self, RowErrors};
use crate::{Containment, Intersection, Interval, IntervalBox, Polytope, SparsePolyZonotope, hcat};

/// `Z = { c + G β | β ∈ [-1, 1]^m }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonotope {
    /// Center `c`; its length is the dimension of the set.
    center: DVector<f64>,
    /// `dim × m` generator matrix; one row per entry of [`center`](Self::center).
    generators: DMatrix<f64>,
}

impl Zonotope {
    /// The zonotope with the given center and generators (one per column).
    ///
    /// # Panics
    ///
    /// Panics if `generators` does not have one row per entry of `center`.
    #[must_use]
    pub fn new(center: DVector<f64>, generators: DMatrix<f64>) -> Self {
        assert_eq!(
            center.len(),
            generators.nrows(),
            "generator rows must match dimension"
        );
        Self { center, generators }
    }

    /// Exact conversion of a box: its midpoint as center, half widths on the diagonal.
    ///
    /// With the `sound` feature the half widths are rounded up, so the result
    /// covers the box even when the midpoint is not representable.
    #[must_use]
    pub fn from_box(b: &IntervalBox) -> Self {
        let (c, r): (Vec<f64>, Vec<f64>) = b.0.iter().map(|i| round::mid_rad(i.lo, i.hi)).unzip();

        Self::new(
            DVector::from_vec(c),
            DMatrix::from_diagonal(&DVector::from_vec(r)),
        )
    }

    /// The zonotope with center `center` and generators `generators`, plus one
    /// axis-aligned generator per nonzero rounding-error bound in `errs`.
    fn with_errors(center: DVector<f64>, generators: &DMatrix<f64>, errs: &RowErrors) -> Self {
        Self::new(center, hcat(generators, &errs.generators()))
    }

    /// Center `c`.
    #[must_use]
    pub fn center(&self) -> &DVector<f64> {
        &self.center
    }

    /// Generator matrix `G`, one generator per column and one row per dimension.
    #[must_use]
    pub fn generators(&self) -> &DMatrix<f64> {
        &self.generators
    }

    /// Dimension of the space the set lives in.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.center.len()
    }

    /// Number of generators `m`, the columns of [`generators`](Self::generators).
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.generators.ncols()
    }

    /// `m / n`, the usual measure of representation size.
    ///
    /// # Panics
    ///
    /// Panics if the set is zero-dimensional, where the order is undefined.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "generator counts and dimensions stay far below 2^52"
    )]
    pub fn order(&self) -> f64 {
        assert!(self.dim() > 0, "order of a zero-dimensional zonotope");
        self.num_generators() as f64 / self.dim() as f64
    }

    /// The point `c + G β` for the factor values `beta`, one per generator.
    ///
    /// # Panics
    ///
    /// Panics if `beta` does not have one entry per generator.
    #[must_use]
    pub fn evaluate(&self, beta: &[f64]) -> DVector<f64> {
        &self.center + &self.generators * DVector::from_column_slice(beta)
    }

    /// Exact: `{ M x | x ∈ Z }`.
    ///
    /// # Panics
    ///
    /// Panics if `m` does not have one column per dimension.
    #[must_use]
    pub fn linear_map(&self, m: &DMatrix<f64>) -> Self {
        let mut errs = RowErrors::new(m.nrows());
        let c = round::mat_vec(m, &self.center, &mut errs);
        let g = round::mat_mul(m, &self.generators, &mut errs);

        Self::with_errors(c, &g, &errs)
    }

    /// Exact: `{ x + v | x ∈ Z }`.
    ///
    /// # Panics
    ///
    /// Panics if `v` does not have the set's dimension.
    #[must_use]
    pub fn translate(&self, v: &DVector<f64>) -> Self {
        let mut errs = RowErrors::new(self.dim());
        let c = round::vec_add(&self.center, v, &mut errs);

        Self::with_errors(c, &self.generators, &errs)
    }

    /// Exact Minkowski sum `{ x + y | x ∈ self, y ∈ o }`: centers add, generators concatenate.
    ///
    /// # Panics
    ///
    /// Panics if the two zonotopes have different dimensions.
    #[must_use]
    pub fn minkowski_sum(&self, o: &Self) -> Self {
        let mut errs = RowErrors::new(self.dim());
        let c = round::vec_add(&self.center, &o.center, &mut errs);

        Self::with_errors(c, &hcat(&self.generators, &o.generators), &errs)
    }

    /// Enclosure with at most `order · dim` generators: the generators with the
    /// smallest `‖g‖₁ − ‖g‖∞` are replaced by a box, as `method` says.
    /// Returns a copy if the zonotope already meets the order.
    ///
    /// With the `sound` feature and [`Reduction::Pca`], up to `dim` generators
    /// for rounding errors come on top.
    ///
    /// # Panics
    ///
    /// Panics if `order < 1`.
    #[must_use]
    pub fn reduce(&self, order: f64, method: Reduction) -> Self {
        let g = &self.generators;
        let Some((kept, dropped)) = reduce::select(g.ncols(), self.dim(), order, |j| {
            reduce::girard_metric(g.column(j).iter())
        }) else {
            return self.clone();
        };

        let mut errs = RowErrors::new(self.dim());
        let dropped = g.select_columns(&dropped);
        let enclosure = reduce::enclose(&dropped, method, &mut errs);

        Self::with_errors(
            self.center.clone(),
            &hcat(&g.select_columns(&kept), &enclosure),
            &errs,
        )
    }

    /// Whether the zonotope meets the polytope `p`; see
    /// [`SparsePolyZonotope::intersection`]. Against a single halfspace the
    /// answer is definite without any splits, up to rounding at the boundary.
    ///
    /// # Panics
    ///
    /// Panics if a facet normal does not have the zonotope's dimension.
    #[must_use]
    pub fn intersection(&self, p: &Polytope, max_splits: usize) -> Intersection {
        SparsePolyZonotope::from_zonotope(self).intersection(p, max_splits)
    }

    /// Whether the zonotope lies inside the polytope `p`; see
    /// [`SparsePolyZonotope::containment`]. The answer is definite without
    /// any splits, up to rounding at the boundary.
    ///
    /// # Panics
    ///
    /// Panics if a facet normal does not have the zonotope's dimension.
    #[must_use]
    pub fn containment(&self, p: &Polytope, max_splits: usize) -> Containment {
        SparsePolyZonotope::from_zonotope(self).containment(p, max_splits)
    }

    /// Vertices of a 2-D zonotope in counter-clockwise order, for drawing.
    ///
    /// Sorts the generators by angle and walks around the boundary, adding
    /// each one twice. Computed in plain `f64`, so it is an outline, not an
    /// enclosure. A zonotope without nonzero generators gives its center.
    ///
    /// # Panics
    ///
    /// Panics if the zonotope is not two-dimensional.
    #[must_use]
    pub fn polygon(&self) -> Vec<[f64; 2]> {
        assert_eq!(self.dim(), 2, "polygon needs a two-dimensional zonotope");
        // Point every generator into the half plane of angles in [0, π), then sort by angle.
        let mut gens: Vec<[f64; 2]> = self
            .generators
            .column_iter()
            .filter(|g| g[0] != 0.0 || g[1] != 0.0)
            .map(|g| {
                if g[1] < 0.0 || (g[1] == 0.0 && g[0] < 0.0) {
                    [-g[0], -g[1]]
                } else {
                    [g[0], g[1]]
                }
            })
            .collect();
        // Within a half plane, a comes first when b lies counter-clockwise of it.
        gens.sort_by(|a, b| {
            let cross = a[0] * b[1] - a[1] * b[0];
            // `>`/`<`, not `total_cmp`, so that -0.0 and 0.0 both count as parallel.
            if cross > 0.0 {
                core::cmp::Ordering::Less
            } else if cross < 0.0 {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        });

        // Start at the lowest vertex, c − Σ g, then go up the right side and down the left.
        let mut p = [self.center[0], self.center[1]];
        for g in &gens {
            p = [p[0] - g[0], p[1] - g[1]];
        }
        let mut vertices = alloc::vec![p];
        for sign in [2.0, -2.0] {
            for g in &gens {
                p = [p[0] + sign * g[0], p[1] + sign * g[1]];
                vertices.push(p);
            }
        }
        if !gens.is_empty() {
            vertices.pop(); // back at the start
        }

        vertices
    }

    /// Tight interval enclosure (rounded outward with the `sound` feature).
    #[must_use]
    pub fn interval_hull(&self) -> IntervalBox {
        IntervalBox(
            (0..self.dim())
                .map(|i| {
                    let r = round::abs_sum_up(self.generators.row(i).iter());
                    Interval::point(self.center[i]) + Interval::new(-r, r)
                })
                .collect(),
        )
    }
}

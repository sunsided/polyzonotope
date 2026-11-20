//! Constrained polynomial zonotopes (Kochdumper & Althoff, 2023): polynomial
//! zonotopes restricted to the factor values where a polynomial constraint
//! vanishes. Closed under intersection, so a cut by an obstacle stays exact.

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::{DMatrix, DVector};

use crate::round;
use crate::{FactorId, Halfspace, Interval, IntervalBox, Polytope, SparsePolyZonotope};

/// `{ x(α, β) | h(α, β) = 0 }` for a polynomial zonotope `x` and a polynomial
/// constraint `h` over the same factors.
///
/// Stored as one stacked polynomial zonotope: the first [`dim`](Self::dim)
/// rows are the set, the remaining rows the constraints. Splits, compaction
/// and rounding-error bookkeeping act on both at once, which keeps them
/// correlated. Under the `sound` feature, rounding errors in constraint rows
/// relax the constraint, which only grows the set.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstrainedPolyZonotope {
    /// Rows `0..dim` are the set, rows `dim..` the constraints.
    stacked: SparsePolyZonotope,
    /// Dimension of the set.
    dim: usize,
}

impl From<SparsePolyZonotope> for ConstrainedPolyZonotope {
    /// The set without constraints.
    fn from(set: SparsePolyZonotope) -> Self {
        Self {
            dim: set.dim(),
            stacked: set,
        }
    }
}

impl ConstrainedPolyZonotope {
    /// `{ set(α, β) | constraints(α, β') = 0 }`. Dependent factors with the
    /// same [`FactorId`] in both are the same factor. Independent factors are
    /// positional, so the `β'` of `constraints` are separate from the `β` of
    /// `set`; build constraints on the set's `β` with
    /// [`intersect_halfspace`](Self::intersect_halfspace) or
    /// [`intersection`](Self::intersection) instead.
    #[must_use]
    pub fn new(set: &SparsePolyZonotope, constraints: &SparsePolyZonotope) -> Self {
        Self {
            stacked: set.cartesian_product(constraints),
            dim: set.dim(),
        }
    }

    /// Dimension of the space the set lives in.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of constraint rows.
    #[must_use]
    pub fn num_constraints(&self) -> usize {
        self.stacked.dim() - self.dim
    }

    /// The set and its constraints as one stacked polynomial zonotope: rows
    /// `0..dim` are the set, the rest must vanish.
    #[must_use]
    pub fn stacked(&self) -> &SparsePolyZonotope {
        &self.stacked
    }

    /// The set with its constraints dropped: an enclosure.
    #[must_use]
    pub fn unconstrained(&self) -> SparsePolyZonotope {
        self.stacked
            .linear_map(&DMatrix::from_fn(self.dim, self.stacked.dim(), |i, j| {
                f64::from(u8::from(i == j))
            }))
    }

    /// Exact: `{ M x | x ∈ CPZ }`.
    ///
    /// # Panics
    ///
    /// Panics if `m` does not have one column per dimension.
    #[must_use]
    pub fn linear_map(&self, m: &DMatrix<f64>) -> Self {
        assert_eq!(
            m.ncols(),
            self.dim,
            "matrix columns must match the dimension"
        );
        let k = self.num_constraints();
        let block = DMatrix::from_fn(m.nrows() + k, self.dim + k, |i, j| {
            match (i < m.nrows(), j < self.dim) {
                (true, true) => m[(i, j)],
                (false, false) => f64::from(u8::from(i - m.nrows() == j - self.dim)),
                _ => 0.0,
            }
        });

        Self {
            stacked: self.stacked.linear_map(&block),
            dim: m.nrows(),
        }
    }

    /// Exact: `{ x + v | x ∈ CPZ }`.
    ///
    /// # Panics
    ///
    /// Panics if `v` does not have the set's dimension.
    #[must_use]
    pub fn translate(&self, v: &DVector<f64>) -> Self {
        assert_eq!(v.len(), self.dim, "translation must match the dimension");
        let padded = DVector::from_fn(
            self.stacked.dim(),
            |i, _| if i < self.dim { v[i] } else { 0.0 },
        );

        Self {
            stacked: self.stacked.translate(&padded),
            dim: self.dim,
        }
    }

    /// Exact Minkowski sum `{ x + y | x ∈ self, y ∈ other }`, the operands
    /// treated as independent.
    ///
    /// # Panics
    ///
    /// Panics if the two sets have different dimensions.
    #[must_use]
    pub fn minkowski_sum(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "sets must have the same dimension");
        let (k1, k2) = (self.num_constraints(), other.num_constraints());
        // Rows: x1 + x2, then the constraints of both.
        let first = self.embed(self.dim + k1 + k2, self.dim);
        let second = other
            .with_fresh_ids()
            .embed(self.dim + k1 + k2, self.dim + k1);

        Self {
            stacked: first.exact_plus(&second),
            dim: self.dim,
        }
    }

    /// Exact intersection `self ∩ other`. Factors shared by the two sets stay
    /// shared.
    ///
    /// # Panics
    ///
    /// Panics if the two sets have different dimensions.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "sets must have the same dimension");
        let (n, k1, k2) = (self.dim, self.num_constraints(), other.num_constraints());
        let total = n + k1 + k2 + n;
        // Rows: x1, the constraints of both, then the new constraint x1 − x2 = 0.
        let first = self.embed_with(
            total,
            |i, row| match row {
                r if r < n => (r == i).then_some(1.0),
                r if r >= n + k1 + k2 => (r - (n + k1 + k2) == i).then_some(1.0),
                _ => None,
            },
            |c, row| (row == n + c).then_some(1.0),
        );
        let second = other.embed_with(
            total,
            |i, row| (row >= n + k1 + k2 && row - (n + k1 + k2) == i).then_some(-1.0),
            |c, row| (row == n + k1 + c).then_some(1.0),
        );

        Self {
            stacked: first.exact_plus(&second),
            dim: n,
        }
    }

    /// Exact intersection with the halfspace `{ x | aᵀ x <= d }`.
    ///
    /// Adds the constraint `aᵀ x + s - d = 0` with a slack `s ∈ [0, u]`, where
    /// `u >= d - min aᵀ x` from the interval hull, carried by a fresh factor.
    /// Skips the constraint if the hull already lies in the halfspace, and
    /// adds the unsatisfiable constraint `1 = 0` if it lies outside.
    ///
    /// # Panics
    ///
    /// Panics if the normal does not have the set's dimension.
    #[must_use]
    pub fn intersect_halfspace(&self, h: &Halfspace) -> Self {
        assert_eq!(h.normal.len(), self.dim, "normal must match the dimension");
        // Append the row aᵀx by one linear map, so it shares every α and β with the set.
        let rows = self.stacked.dim();
        let extended = self
            .stacked
            .linear_map(&DMatrix::from_fn(rows + 1, rows, |i, j| {
                if i < rows {
                    f64::from(u8::from(i == j))
                } else if j < self.dim {
                    h.normal[j]
                } else {
                    0.0
                }
            }));
        let range = extended.interval_hull().0[rows];
        if range.hi <= h.offset {
            return self.clone();
        }

        let last = |v: f64| DVector::from_fn(rows + 1, |i, _| if i == rows { v } else { 0.0 });
        if range.lo > h.offset {
            // The unsatisfiable constraint 1 = 0: zero the new row, then move it to 1.
            let zeroed = DMatrix::from_fn(rows + 1, rows, |i, j| f64::from(u8::from(i == j)));
            return Self {
                stacked: self.stacked.linear_map(&zeroed).translate(&last(1.0)),
                dim: self.dim,
            };
        }

        // aᵀx + s − d = 0 with s = half + half · α_s over [0, 2 half] ⊇ [0, d − min aᵀx].
        let width = round::add_up(h.offset, -range.lo);
        let mut half = 0.5 * width;
        if 2.0 * half < width {
            // Halving rounded down (`width / 2` is subnormal); doubling is exact.
            half = half.next_up();
        }
        let slack = SparsePolyZonotope::new(
            last(-h.offset),
            DMatrix::from_fn(rows + 1, 2, |i, _| if i == rows { half } else { 0.0 }),
            DMatrix::zeros(rows + 1, 0),
            DMatrix::from_row_slice(1, 2, &[0, 1]),
            vec![FactorId::fresh()],
        )
        .expect("consistent shapes");

        Self {
            stacked: extended.exact_plus(&slack),
            dim: self.dim,
        }
    }

    /// Exact intersection with every facet of `p`.
    ///
    /// # Panics
    ///
    /// Panics if a facet normal does not have the set's dimension.
    #[must_use]
    pub fn intersect_polytope(&self, p: &Polytope) -> Self {
        p.facets
            .iter()
            .fold(self.clone(), |acc, h| acc.intersect_halfspace(h))
    }

    /// Interval enclosure that ignores the constraints: sound, but can be loose.
    #[must_use]
    pub fn interval_hull(&self) -> IntervalBox {
        IntervalBox(self.stacked.interval_hull().0[..self.dim].to_vec())
    }

    /// Interval enclosure refined by splitting, `None` if the set is proven empty.
    ///
    /// Splits the stacked set up to `max_splits` times, always the widest
    /// piece that might be feasible, and drops every piece whose constraint
    /// hull excludes zero. The result is the hull of the remaining pieces.
    #[must_use]
    pub fn interval_hull_split(&self, max_splits: usize) -> Option<IntervalBox> {
        let pieces = self.feasible_pieces(max_splits);
        let hulls: Vec<IntervalBox> = pieces
            .iter()
            .map(SparsePolyZonotope::interval_hull)
            .collect();
        let first = hulls.first()?;

        Some(IntervalBox(
            (0..self.dim)
                .map(|i| {
                    let lo = hulls
                        .iter()
                        .map(|h| h.0[i].lo)
                        .fold(first.0[i].lo, f64::min);
                    let hi = hulls
                        .iter()
                        .map(|h| h.0[i].hi)
                        .fold(first.0[i].hi, f64::max);
                    Interval::new(lo, hi)
                })
                .collect(),
        ))
    }

    /// Whether splitting at most `max_splits` times proves the set empty.
    /// `false` means "not proven", not "non-empty".
    #[must_use]
    pub fn proven_empty(&self, max_splits: usize) -> bool {
        self.feasible_pieces(max_splits).is_empty()
    }

    /// Pieces of the stacked set that might satisfy the constraints, after at
    /// most `max_splits` splits of the widest one.
    fn feasible_pieces(&self, max_splits: usize) -> Vec<SparsePolyZonotope> {
        let width = |pz: &SparsePolyZonotope| {
            pz.interval_hull()
                .0
                .iter()
                .map(Interval::width)
                .sum::<f64>()
        };
        let mut pieces: Vec<SparsePolyZonotope> = vec![self.stacked.clone()];
        for _ in 0..max_splits {
            pieces.retain(|p| self.might_be_feasible(p));
            let candidates =
                (0..pieces.len()).filter_map(|i| Some((i, pieces[i].heaviest_factor_overall()?)));
            let Some((i, factor)) =
                candidates.max_by(|a, b| width(&pieces[a.0]).total_cmp(&width(&pieces[b.0])))
            else {
                break;
            };
            let (lower, upper) = pieces.swap_remove(i).split(factor);
            pieces.push(lower);
            pieces.push(upper);
        }

        pieces.retain(|p| self.might_be_feasible(p));
        pieces
    }

    /// Whether the hull of every constraint row of `piece` contains zero.
    ///
    /// Plain mode allows a rounding residue relative to the magnitude of the
    /// whole piece: a constraint like `x − x = 0` built along two paths can
    /// leave a few ulps after its terms cancel, and dropping that piece would
    /// call a non-empty set empty. Under `sound` the residue is already an
    /// error generator.
    fn might_be_feasible(&self, piece: &SparsePolyZonotope) -> bool {
        let hull = piece.interval_hull();
        let slack = if cfg!(feature = "sound") {
            0.0
        } else {
            let magnitude = piece
                .center()
                .iter()
                .chain(piece.dependent_generators().iter())
                .chain(piece.independent_generators().iter());
            1e3 * f64::EPSILON * round::abs_sum_up(magnitude)
        };

        hull.0[self.dim..]
            .iter()
            .all(|iv| iv.lo <= slack && -slack <= iv.hi)
    }

    /// Same set, with all dependent factors renamed to fresh ids.
    fn with_fresh_ids(&self) -> Self {
        Self {
            stacked: self.stacked.with_fresh_ids(),
            dim: self.dim,
        }
    }

    /// The stacked set mapped into `total` rows: set rows stay in place and
    /// constraint row `r` moves to row `first_constraint + r`.
    fn embed(&self, total: usize, first_constraint: usize) -> SparsePolyZonotope {
        self.embed_with(
            total,
            |i, row| (i == row).then_some(1.0),
            |c, row| (first_constraint + c == row).then_some(1.0),
        )
    }

    /// The stacked set mapped into `total` rows by the matrix whose entry for
    /// target `row` and set row `i` is `set_weight(i, row)`, and for constraint
    /// row `c` is `constraint_weight(c, row)`; `None` means zero.
    fn embed_with(
        &self,
        total: usize,
        set_weight: impl Fn(usize, usize) -> Option<f64>,
        constraint_weight: impl Fn(usize, usize) -> Option<f64>,
    ) -> SparsePolyZonotope {
        let m = DMatrix::from_fn(total, self.stacked.dim(), |row, j| {
            let w = if j < self.dim {
                set_weight(j, row)
            } else {
                constraint_weight(j - self.dim, row)
            };
            w.unwrap_or(0.0)
        });

        self.stacked.linear_map(&m)
    }
}

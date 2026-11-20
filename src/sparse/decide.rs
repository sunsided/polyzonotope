//! Three-valued intersection and containment checks against polytopes.

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::DMatrix;

use crate::{Containment, FactorId, Halfspace, Intersection, Polytope, SparsePolyZonotope};

impl SparsePolyZonotope {
    /// Whether the set meets the polytope `p`, refined by splitting.
    ///
    /// A piece is discarded when the sound lower bound of `aᵀ x` over it is
    /// strictly above `d` for some facet `aᵀ x <= d`. A piece whose witness
    /// point lies in `p` gives [`Intersection::Intersects`]; any other piece is
    /// split along its heaviest factor. At most `max_splits` splits are made
    /// in total. Factors with an exponent above the limit of
    /// [`split`](Self::split) are never split; such pieces give `Unknown`.
    ///
    /// # Panics
    ///
    /// Panics if a facet normal does not have the set's dimension.
    #[must_use]
    pub fn intersection(&self, p: &Polytope, max_splits: usize) -> Intersection {
        let mut pieces = vec![self.clone()];
        let mut splits = 0;
        while let Some(piece) = pieces.pop() {
            if p.facets.iter().any(|h| piece.lies_beyond(h)) {
                continue;
            }

            if piece.witnesses(p).iter().any(|x| p.contains(x, 0.0)) {
                return Intersection::Intersects;
            }

            let Some(factor) = piece.heaviest_factor_overall() else {
                return Intersection::Unknown;
            };
            if splits == max_splits {
                return Intersection::Unknown;
            }

            splits += 1;
            let (lower, upper) = piece.split(factor);
            pieces.push(lower);
            pieces.push(upper);
        }

        Intersection::Disjoint
    }

    /// Whether the set lies inside the polytope `p`, refined by splitting.
    ///
    /// For each facet `aᵀ x <= d`, branch and bound on `aᵀ x` either proves
    /// the maximum is at most `d`, or finds a point above `d`. At most
    /// `max_splits` splits are made in total. Factors with an exponent above
    /// the limit of [`split`](Self::split) are never split.
    ///
    /// # Panics
    ///
    /// Panics if a facet normal does not have the set's dimension.
    #[must_use]
    pub fn containment(&self, p: &Polytope, max_splits: usize) -> Containment {
        let mut budget = max_splits;
        let mut verdict = Containment::Inside;
        for h in &p.facets {
            let d = h.offset;
            let along = self.along(h);
            let search =
                along.search_upper_bound(0, 1.0, budget, |outer, inner| outer <= d || inner > d);
            budget -= search.splits;
            if search.inner > d {
                return Containment::NotInside;
            }

            if search.outer > d {
                verdict = Containment::Unknown;
            }
        }

        verdict
    }

    /// Pieces whose union is the set, from `splits` splits of the widest
    /// piece along its heaviest factor; fewer if no piece has a factor left.
    ///
    /// Each piece is exact. Their zonotope enclosures together hug the set
    /// much tighter than [`to_zonotope`](Self::to_zonotope) alone, which is
    /// what plots of curved sets use.
    #[must_use]
    pub fn split_pieces(&self, splits: usize) -> Vec<Self> {
        let width = |pz: &Self| {
            pz.interval_hull()
                .0
                .iter()
                .map(crate::Interval::width)
                .sum::<f64>()
        };
        let mut pieces = vec![self.clone()];
        for _ in 0..splits {
            let splittable =
                (0..pieces.len()).filter_map(|i| Some((i, pieces[i].heaviest_factor_overall()?)));
            let Some((i, factor)) =
                splittable.max_by(|a, b| width(&pieces[a.0]).total_cmp(&width(&pieces[b.0])))
            else {
                break;
            };
            let (lower, upper) = pieces.swap_remove(i).split(factor);
            pieces.push(lower);
            pieces.push(upper);
        }

        pieces
    }

    /// The 1-D set `{ aᵀ x | x ∈ self }` for the facet normal `a`.
    fn along(&self, h: &Halfspace) -> Self {
        self.linear_map(&DMatrix::from_row_slice(
            1,
            h.normal.len(),
            h.normal.as_slice(),
        ))
    }

    /// Whether every point of the set lies strictly outside `h`, from the
    /// sound lower bound of `aᵀ x`.
    fn lies_beyond(&self, h: &Halfspace) -> bool {
        self.along(h).interval_hull().0[0].lo > h.offset
    }

    /// Candidate points of the set inside `p`: the center (all factors zero)
    /// and, per facet, the corner that minimises `aᵀ x` for a zonotope.
    fn witnesses(&self, p: &Polytope) -> Vec<nalgebra::DVector<f64>> {
        let zeros = (
            vec![0.0; self.ids().len()],
            vec![0.0; self.independent_generators().ncols()],
        );
        let mut points = vec![self.evaluate(&zeros.0, &zeros.1)];
        for h in &p.facets {
            let (alpha, beta) = self.descent_corner(&h.normal);
            points.push(self.evaluate(&alpha, &beta));
        }

        points
    }

    /// Factor values that push `aᵀ x` down: each `β` and each `α_k` that
    /// appears alone with an odd power at the sign opposite to its pull.
    fn descent_corner(&self, a: &nalgebra::DVector<f64>) -> (Vec<f64>, Vec<f64>) {
        let down = |x: f64| {
            if x > 0.0 {
                -1.0
            } else if x < 0.0 {
                1.0
            } else {
                0.0
            }
        };
        let (g, e) = (self.dependent_generators(), self.exponents());
        let alpha = (0..self.ids().len())
            .map(|k| {
                let alone_odd = |j: usize| {
                    e[(k, j)] % 2 == 1 && (0..e.nrows()).all(|r| r == k || e[(r, j)] == 0)
                };
                down(
                    (0..g.ncols())
                        .filter(|&j| alone_odd(j))
                        .map(|j| a.dot(&g.column(j)))
                        .sum(),
                )
            })
            .collect();
        let beta = self
            .independent_generators()
            .column_iter()
            .map(|gi| down(a.dot(&gi)))
            .collect();

        (alpha, beta)
    }

    /// The factor whose monomials weigh most over all dimensions, by
    /// `Σ_j ‖G[:, j]‖₁ E[k, j]`; `None` if no monomial depends on a factor
    /// that [`split`](Self::split) accepts.
    fn heaviest_factor_overall(&self) -> Option<FactorId> {
        let (g, e) = (self.dependent_generators(), self.exponents());
        let weight = |k: usize| -> f64 {
            (0..g.ncols())
                .map(|j| g.column(j).lp_norm(1) * f64::from(e[(k, j)]))
                .sum()
        };

        (0..self.ids().len())
            .map(|k| (k, weight(k)))
            .filter(|&(k, w)| w > 0.0 && self.can_split(k))
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(k, _)| self.ids()[k])
    }
}

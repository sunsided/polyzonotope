//! Interval hulls tightened by recursive splitting.

use alloc::vec;
use alloc::vec::Vec;

use crate::{Interval, IntervalBox, SparsePolyZonotope};

/// A piece of the set during branch and bound, with its outer bound.
struct Piece {
    /// The piece itself; the pieces together cover the original set.
    set: SparsePolyZonotope,
    /// Upper bound of `sign · x_dim` over the piece, from its interval hull.
    outer: f64,
}

impl SparsePolyZonotope {
    /// Interval enclosure tightened by splitting the dependent factors.
    ///
    /// For each dimension and each bound, the piece that attains the bound is
    /// split along the factor that weighs most in that dimension, until the
    /// bound is within `tol` of a point of the set, or `max_splits` splits per
    /// bound are spent. The result is always sound and never wider than
    /// [`interval_hull`](Self::interval_hull); `tol` and `max_splits` only
    /// trade tightness for time.
    #[must_use]
    pub fn interval_hull_split(&self, tol: f64, max_splits: usize) -> IntervalBox {
        let coarse = self.interval_hull();
        IntervalBox(
            (0..self.dim())
                .map(|i| {
                    let hi = self.tight_upper_bound(i, 1.0, tol, max_splits);
                    let lo = -self.tight_upper_bound(i, -1.0, tol, max_splits);
                    let (lo, hi) = (lo.max(coarse.0[i].lo), hi.min(coarse.0[i].hi));
                    // Plain-mode rounding can cross the bounds of a near-degenerate set.
                    if lo <= hi {
                        Interval::new(lo, hi)
                    } else {
                        coarse.0[i]
                    }
                })
                .collect(),
        )
    }

    /// Upper bound of `sign · x_dim` over the set, by branch and bound.
    fn tight_upper_bound(&self, dim: usize, sign: f64, tol: f64, max_splits: usize) -> f64 {
        let mut inner = self.best_point_value(dim, sign);
        let mut pieces = vec![Piece {
            outer: self.signed_hull_bound(dim, sign),
            set: self.clone(),
        }];
        for _ in 0..max_splits {
            let worst = arg_max(&pieces);
            if pieces[worst].outer - inner <= tol {
                break;
            }

            let Some(factor) = pieces[worst].set.heaviest_factor(dim) else {
                // No dependent factor left: the hull of this piece is exact in `dim`.
                break;
            };
            let piece = pieces.swap_remove(worst);
            let (lower, upper) = piece.set.split(factor);
            for set in [lower, upper] {
                inner = inner.max(set.best_point_value(dim, sign));
                let outer = set.signed_hull_bound(dim, sign).min(piece.outer);
                pieces.push(Piece { set, outer });
            }
        }

        pieces
            .iter()
            .map(|p| p.outer)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Upper bound of `sign · x_dim` from the interval hull.
    fn signed_hull_bound(&self, dim: usize, sign: f64) -> f64 {
        let iv = self.interval_hull().0[dim];
        if sign > 0.0 { iv.hi } else { -iv.lo }
    }

    /// `sign · x_dim` at a point of the set: all `α = 0`, and each `β` at the
    /// sign that pushes `sign · x_dim` up. A lower bound of the true maximum.
    fn best_point_value(&self, dim: usize, sign: f64) -> f64 {
        let alpha = vec![0.0; self.ids().len()];
        let beta: Vec<f64> = self
            .independent_generators()
            .row(dim)
            .iter()
            .map(|&g| if sign * g >= 0.0 { 1.0 } else { -1.0 })
            .collect();

        sign * self.evaluate(&alpha, &beta)[dim]
    }

    /// The factor whose monomials weigh most in dimension `dim`, by
    /// `Σ_j |G[dim, j]| E[k, j]`; `None` if no monomial depends on a factor.
    fn heaviest_factor(&self, dim: usize) -> Option<crate::FactorId> {
        let (g, e) = (self.dependent_generators(), self.exponents());
        let weight = |k: usize| -> f64 {
            (0..g.ncols())
                .map(|j| g[(dim, j)].abs() * f64::from(e[(k, j)]))
                .sum()
        };

        (0..self.ids().len())
            .map(|k| (k, weight(k)))
            .filter(|&(_, w)| w > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(k, _)| self.ids()[k])
    }
}

/// Index of the piece with the largest outer bound; `pieces` is never empty.
fn arg_max(pieces: &[Piece]) -> usize {
    (0..pieces.len())
        .max_by(|&a, &b| pieces[a].outer.total_cmp(&pieces[b].outer))
        .expect("branch and bound keeps at least one piece")
}

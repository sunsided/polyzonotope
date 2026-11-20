//! Interval hulls tightened by recursive splitting.

use alloc::vec;
use alloc::vec::Vec;

use crate::{Interval, IntervalBox, SparsePolyZonotope};

/// Result of [`SparsePolyZonotope::search_upper_bound`].
pub(super) struct Search {
    /// Sound upper bound of the maximum.
    pub(super) outer: f64,
    /// Value at a point of the set, so at most the maximum (plain `f64`).
    pub(super) inner: f64,
    /// Number of splits spent.
    pub(super) splits: usize,
}

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
    /// trade tightness for time. Factors with an exponent above the limit of
    /// [`split`](Self::split) are never split, so this never panics.
    #[must_use]
    pub fn interval_hull_split(&self, tol: f64, max_splits: usize) -> IntervalBox {
        let coarse = self.interval_hull();
        IntervalBox(
            (0..self.dim())
                .map(|i| {
                    let close_enough = |outer: f64, inner: f64| outer - inner <= tol;
                    let hi = self
                        .search_upper_bound(i, 1.0, max_splits, close_enough)
                        .outer;
                    let lo = -self
                        .search_upper_bound(i, -1.0, max_splits, close_enough)
                        .outer;
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

    /// Bounds on `max sign · x_dim` over the set, by branch and bound: splits
    /// the piece with the largest outer bound until `stop(outer, inner)` holds
    /// or `max_splits` splits are spent.
    pub(super) fn search_upper_bound(
        &self,
        dim: usize,
        sign: f64,
        max_splits: usize,
        stop: impl Fn(f64, f64) -> bool,
    ) -> Search {
        let mut inner = self.best_point_value(dim, sign);
        let mut pieces = vec![Piece {
            outer: self.signed_hull_bound(dim, sign),
            set: self.clone(),
        }];
        let mut splits = 0;
        while splits < max_splits {
            let worst = arg_max(&pieces);
            if stop(pieces[worst].outer, inner) {
                break;
            }

            let Some(factor) = pieces[worst].set.heaviest_factor(dim) else {
                // No dependent factor left: the hull of this piece is exact in `dim`.
                break;
            };
            let piece = pieces.swap_remove(worst);
            splits += 1;
            let (lower, upper) = piece.set.split(factor);
            for set in [lower, upper] {
                inner = inner.max(set.best_point_value(dim, sign));
                let outer = set.signed_hull_bound(dim, sign).min(piece.outer);
                pieces.push(Piece { set, outer });
            }
        }

        Search {
            outer: pieces[arg_max(&pieces)].outer,
            inner,
            splits,
        }
    }

    /// Upper bound of `sign · x_dim` from the interval hull.
    fn signed_hull_bound(&self, dim: usize, sign: f64) -> f64 {
        let iv = self.interval_hull().0[dim];
        if sign > 0.0 { iv.hi } else { -iv.lo }
    }

    /// The larger `sign · x_dim` of two points of the set, a lower bound of the
    /// true maximum. Both set each `β` to the sign that pushes `sign · x_dim`
    /// up. One has every `α = 0`; the other sets each `α_k` to the sign of the
    /// monomials that are odd powers of `α_k` alone, which is the exact
    /// maximiser for a set that came from a zonotope.
    ///
    /// Evaluated in plain `f64`, so the bound is not certified.
    fn best_point_value(&self, dim: usize, sign: f64) -> f64 {
        let up = |x: f64| if sign * x >= 0.0 { 1.0 } else { -1.0 };
        let beta: Vec<f64> = self
            .independent_generators()
            .row(dim)
            .iter()
            .map(|&g| up(g))
            .collect();
        let (g, e) = (self.dependent_generators(), self.exponents());
        let corner: Vec<f64> = (0..self.ids().len())
            .map(|k| {
                let alone_odd = |j: usize| {
                    e[(k, j)] % 2 == 1 && (0..e.nrows()).all(|r| r == k || e[(r, j)] == 0)
                };
                let pull: f64 = (0..g.ncols())
                    .filter(|&j| alone_odd(j))
                    .map(|j| g[(dim, j)])
                    .sum();
                if pull == 0.0 { 0.0 } else { up(pull) }
            })
            .collect();
        let center = vec![0.0; self.ids().len()];

        [center, corner]
            .iter()
            .map(|alpha| sign * self.evaluate(alpha, &beta)[dim])
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The factor whose monomials weigh most in dimension `dim`, by
    /// `Σ_j |G[dim, j]| E[k, j]`; `None` if no monomial depends on a factor
    /// that [`split`](Self::split) accepts.
    fn heaviest_factor(&self, dim: usize) -> Option<crate::FactorId> {
        let (g, e) = (self.dependent_generators(), self.exponents());
        let weight = |k: usize| -> f64 {
            (0..g.ncols())
                .map(|j| g[(dim, j)].abs() * f64::from(e[(k, j)]))
                .sum()
        };

        (0..self.ids().len())
            .map(|k| (k, weight(k)))
            .filter(|&(k, w)| w > 0.0 && self.can_split(k))
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

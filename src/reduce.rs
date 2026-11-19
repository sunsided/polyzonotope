//! Order reduction: replace the least important generators by a small enclosure.

use alloc::vec::Vec;
use core::cmp::Ordering;

use nalgebra::{DMatrix, DVector};

use crate::round::{self, RowErrors};

/// How [`Zonotope::reduce`](crate::Zonotope::reduce) and
/// [`SparsePolyZonotope::reduce`](crate::SparsePolyZonotope::reduce) enclose
/// the generators they drop.
///
/// Both pick the generators to drop the same way: those with the smallest
/// `‖g‖₁ − ‖g‖∞`, which are closest to axis-aligned and lose the least when
/// boxed (Girard 2005).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Encloses the dropped generators in an axis-aligned box (Girard 2005).
    Girard,
    /// Encloses them in a box aligned with their principal axes, the
    /// eigenvectors of `G Gᵀ` (Kopetzki, Schürmann & Althoff 2017). Tighter
    /// for strongly correlated generators, at the cost of an eigendecomposition.
    Pca,
}

/// Generator indices to keep and to drop, so that at most `order · n`
/// generators remain after the dropped ones become `n` box generators.
///
/// Returns `None` if the `count` generators already meet the order.
///
/// # Panics
///
/// Panics if `order < 1`, which no enclosure with `n` box generators can meet.
pub(crate) fn select(
    count: usize,
    dim: usize,
    order: f64,
    metric: impl Fn(usize) -> f64,
) -> Option<(Vec<usize>, Vec<usize>)> {
    assert!(
        order >= 1.0,
        "the reduced order must be at least 1, got {order}"
    );
    let budget = order * as_f64(dim);
    if as_f64(count) <= budget {
        return None;
    }

    let keep = floor_count(budget) - dim;
    let mut by_metric: Vec<(usize, f64)> = (0..count).map(|j| (j, metric(j))).collect();
    by_metric.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let (kept, dropped) = by_metric.split_at(keep);
    let indices = |part: &[(usize, f64)]| {
        let mut v: Vec<usize> = part.iter().map(|&(j, _)| j).collect();
        v.sort_unstable();
        v
    };

    Some((indices(kept), indices(dropped)))
}

/// Girard's metric `‖g‖₁ − ‖g‖∞`: zero for an axis-aligned generator.
pub(crate) fn girard_metric<'a>(g: impl Iterator<Item = &'a f64>) -> f64 {
    let (sum, max) = g.fold((0.0, 0.0f64), |(s, m), x| (s + x.abs(), m.max(x.abs())));
    sum - max
}

/// Generators of an enclosure of `{ G β | β ∈ [-1, 1]^m }`, with rounding
/// errors recorded in `errs`.
///
/// For [`Reduction::Girard`] the box radii go straight into `errs`, so the box
/// and the rounding errors share one set of axis-aligned generators.
pub(crate) fn enclose(g: &DMatrix<f64>, method: Reduction, errs: &mut RowErrors) -> DMatrix<f64> {
    let n = g.nrows();
    match method {
        Reduction::Girard => {
            for i in 0..n {
                errs.record(i, round::abs_sum_up(g.row(i).iter()));
            }
            DMatrix::zeros(n, 0)
        }
        Reduction::Pca => pca_box(g, errs),
    }
}

/// The box aligned with the principal axes `U` of `G Gᵀ`: `U diag(r)` with
/// `r` the row sums of `|Uᵀ G|`.
///
/// `x = U Uᵀ x + (I − U Uᵀ) x` for every `x`; the first part lies in the box,
/// the second is bounded by `|I − U Uᵀ| |G| 1`, which is zero in plain mode
/// and covers the non-orthogonality of the computed `U` in sound mode.
fn pca_box(g: &DMatrix<f64>, errs: &mut RowErrors) -> DMatrix<f64> {
    let n = g.nrows();
    let u = (g * g.transpose()).symmetric_eigen().eigenvectors;
    let mut rotated_errs = RowErrors::new(n);
    let rotated = round::mat_mul(&u.transpose(), g, &mut rotated_errs);
    let reach = DVector::from_fn(n, |k, _| round::abs_sum_up(g.row(k).iter()));
    for (i, bound) in round::residual_bound(&u, &reach).iter().enumerate() {
        errs.record(i, *bound);
    }

    let columns: Vec<DVector<f64>> = (0..n)
        .map(|i| {
            let r = round::add_up(
                round::abs_sum_up(rotated.row(i).iter()),
                rotated_errs.bound(i),
            );
            round::scaled(&u.column(i).into_owned(), r, errs)
        })
        .filter(|c| c.iter().any(|&x| x != 0.0))
        .collect();

    crate::from_column_list(n, &columns)
}

/// A dimension or generator count as `f64`.
#[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^52")]
fn as_f64(n: usize) -> f64 {
    n as f64
}

/// `⌊x⌋` as a count, for `x >= 0`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "x is a non-negative generator budget far below usize::MAX"
)]
fn floor_count(x: f64) -> usize {
    x as usize
}

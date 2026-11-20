//! Halfspaces and convex polytopes, the obstacles and safe regions sets are checked against.

use alloc::vec;
use alloc::vec::Vec;

use nalgebra::DVector;

use crate::IntervalBox;

/// The closed halfspace `{ x | normalᵀ x <= offset }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Halfspace {
    /// Outward normal; its length is the dimension.
    pub normal: DVector<f64>,
    /// Offset `d` in `normalᵀ x <= d`.
    pub offset: f64,
}

/// Convex polytope `{ x | aᵢᵀ x <= dᵢ for every facet i }`, the intersection
/// of its facet halfspaces; no facets give the whole space.
#[derive(Clone, Debug, PartialEq)]
pub struct Polytope {
    /// Facet halfspaces.
    pub facets: Vec<Halfspace>,
}

impl Polytope {
    /// The box as a polytope: two facets per dimension.
    #[must_use]
    pub fn from_box(b: &IntervalBox) -> Self {
        let n = b.dim();
        let axis = |i: usize, s: f64| DVector::from_fn(n, |k, _| if k == i { s } else { 0.0 });
        let facets =
            b.0.iter()
                .enumerate()
                .flat_map(|(i, iv)| {
                    vec![
                        Halfspace {
                            normal: axis(i, 1.0),
                            offset: iv.hi,
                        },
                        Halfspace {
                            normal: axis(i, -1.0),
                            offset: -iv.lo,
                        },
                    ]
                })
                .collect();

        Self { facets }
    }

    /// Whether `p` satisfies every facet, each relaxed by `tol`.
    #[must_use]
    pub fn contains(&self, p: &DVector<f64>, tol: f64) -> bool {
        self.facets
            .iter()
            .all(|h| h.normal.dot(p) <= h.offset + tol)
    }
}

impl From<Halfspace> for Polytope {
    fn from(h: Halfspace) -> Self {
        Self { facets: vec![h] }
    }
}

/// Answer to "does the set meet the polytope?".
///
/// Only [`Disjoint`](Self::Disjoint) is a proof. Treat
/// [`Intersects`](Self::Intersects) and [`Unknown`](Self::Unknown) alike when
/// safety is at stake, e.g. by rejecting a planned edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intersection {
    /// Proven from sound outer bounds: no point of the set lies in the polytope.
    Disjoint,
    /// A point of the set was found in the polytope. The point is evaluated in
    /// plain `f64`, so this is not certified near the boundary.
    Intersects,
    /// The split budget ran out, or the remaining piece has nothing left to split.
    Unknown,
}

/// Answer to "does the set lie inside the polytope?".
///
/// Only [`Inside`](Self::Inside) is a proof; treat
/// [`NotInside`](Self::NotInside) and [`Unknown`](Self::Unknown) alike when
/// safety is at stake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Containment {
    /// Proven from sound outer bounds: every point of the set satisfies every facet.
    Inside,
    /// A point of the set was found outside a facet. The point is evaluated in
    /// plain `f64`, so this is not certified near the boundary.
    NotInside,
    /// The split budget ran out, or the remaining piece has nothing left to split.
    Unknown,
}

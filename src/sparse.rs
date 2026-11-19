use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::{DMatrix, DVector};

use crate::{Interval, IntervalBox, Zonotope, hcat};

/// Global identifier of a dependent factor `α_k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FactorId(u64);

static NEXT_FACTOR: AtomicU64 = AtomicU64::new(0);

impl FactorId {
    pub fn fresh() -> Self {
        Self(NEXT_FACTOR.fetch_add(1, Ordering::Relaxed))
    }
    pub fn fresh_n(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::fresh()).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    GeneratorRows,
    IndependentRows,
    ExponentColumns,
    ExponentRows,
    DuplicateId,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::GeneratorRows => "dependent generator rows must equal the dimension",
            Self::IndependentRows => "independent generator rows must equal the dimension",
            Self::ExponentColumns => "exponent matrix needs one column per dependent generator",
            Self::ExponentRows => "exponent matrix needs one row per factor id",
            Self::DuplicateId => "factor ids must be distinct",
        })
    }
}

impl std::error::Error for ShapeError {}

/// Sparse polynomial zonotope `⟨c, G, GI, E, id⟩`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparsePolyZonotope {
    c: DVector<f64>,
    g: DMatrix<f64>,
    gi: DMatrix<f64>,
    e: DMatrix<u32>,
    id: Vec<FactorId>,
}

/// How a dependent monomial ranges over `α ∈ [-1, 1]^p`.
enum MonomialRange {
    Constant,  // all exponents zero: exactly 1
    UnitRange, // all exponents even: [0, 1]
    Symmetric, // otherwise: [-1, 1]
}

impl SparsePolyZonotope {
    pub fn new(
        c: DVector<f64>,
        g: DMatrix<f64>,
        gi: DMatrix<f64>,
        e: DMatrix<u32>,
        id: Vec<FactorId>,
    ) -> Result<Self, ShapeError> {
        let n = c.len();
        if g.nrows() != n {
            return Err(ShapeError::GeneratorRows);
        }
        if gi.nrows() != n {
            return Err(ShapeError::IndependentRows);
        }
        if e.ncols() != g.ncols() {
            return Err(ShapeError::ExponentColumns);
        }
        if e.nrows() != id.len() {
            return Err(ShapeError::ExponentRows);
        }
        let mut sorted = id.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(ShapeError::DuplicateId);
        }
        Ok(Self { c, g, gi, e, id })
    }

    /// Exact conversion: every generator becomes a dependent factor.
    pub fn from_zonotope(z: &Zonotope) -> Self {
        let m = z.num_generators();
        Self {
            c: z.center.clone(),
            g: z.generators.clone(),
            gi: DMatrix::zeros(z.dim(), 0),
            e: DMatrix::identity(m, m),
            id: FactorId::fresh_n(m),
        }
    }

    pub fn dim(&self) -> usize {
        self.c.len()
    }
    pub fn center(&self) -> &DVector<f64> {
        &self.c
    }
    pub fn dependent_generators(&self) -> &DMatrix<f64> {
        &self.g
    }
    pub fn independent_generators(&self) -> &DMatrix<f64> {
        &self.gi
    }
    pub fn exponents(&self) -> &DMatrix<u32> {
        &self.e
    }
    pub fn ids(&self) -> &[FactorId] {
        &self.id
    }

    /// Point of the set for the given factor values (`alpha` looked up by
    /// id, `beta` positional).
    pub fn evaluate_with(&self, alpha: impl Fn(FactorId) -> f64, beta: &[f64]) -> DVector<f64> {
        assert_eq!(
            beta.len(),
            self.gi.ncols(),
            "one β per independent generator"
        );
        let a: Vec<f64> = self.id.iter().map(|&k| alpha(k)).collect();
        let mut p = &self.c + &self.gi * DVector::from_column_slice(beta);
        for i in 0..self.g.ncols() {
            let m: f64 = (0..a.len())
                .map(|k| a[k].powi(self.e[(k, i)] as i32))
                .product();
            p += self.g.column(i) * m;
        }
        p
    }

    /// Positional variant of [`evaluate_with`](Self::evaluate_with): `alpha[k]` belongs to `ids()[k]`.
    pub fn evaluate(&self, alpha: &[f64], beta: &[f64]) -> DVector<f64> {
        assert_eq!(alpha.len(), self.id.len(), "one α per factor");
        let pos: HashMap<FactorId, usize> =
            self.id.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        self.evaluate_with(|k| alpha[pos[&k]], beta)
    }

    /// Exact: `{ M x | x ∈ PZ }`.
    pub fn linear_map(&self, m: &DMatrix<f64>) -> Self {
        Self {
            c: m * &self.c,
            g: m * &self.g,
            gi: m * &self.gi,
            ..self.clone()
        }
    }

    pub fn translate(&self, v: &DVector<f64>) -> Self {
        Self {
            c: &self.c + v,
            ..self.clone()
        }
    }

    /// Same set, with all dependent factors renamed to fresh ids (decorrelates it from every other set).
    pub fn with_fresh_ids(&self) -> Self {
        Self {
            id: FactorId::fresh_n(self.id.len()),
            ..self.clone()
        }
    }

    /// Exact Minkowski sum `{ x + y | x ∈ self, y ∈ other }` (sets treated as independent).
    pub fn minkowski_sum(&self, other: &Self) -> Self {
        self.exact_plus(&other.with_fresh_ids())
    }

    /// Exact addition respecting shared factors: `{ p₁(α) + p₂(α) }`.
    pub fn exact_plus(&self, other: &Self) -> Self {
        let (id, e1, e2) = merge_ids(self, other);
        Self {
            c: &self.c + &other.c,
            g: hcat(&self.g, &other.g),
            gi: hcat(&self.gi, &other.gi),
            e: hcat(&e1, &e2),
            id,
        }
        .compact()
    }

    /// Exact Cartesian product, keeping shared factors correlated.
    pub fn cartesian_product(&self, other: &Self) -> Self {
        let (id, e1, e2) = merge_ids(self, other);
        let (n1, n2) = (self.dim(), other.dim());
        let block = |a: &DMatrix<f64>, b: &DMatrix<f64>| {
            DMatrix::from_fn(n1 + n2, a.ncols() + b.ncols(), |i, j| {
                match (i < n1, j < a.ncols()) {
                    (true, true) => a[(i, j)],
                    (false, false) => b[(i - n1, j - a.ncols())],
                    _ => 0.0,
                }
            })
        };
        Self {
            c: DVector::from_iterator(n1 + n2, self.c.iter().chain(other.c.iter()).copied()),
            g: block(&self.g, &other.g),
            gi: block(&self.gi, &other.gi),
            e: hcat(&e1, &e2),
            id,
        }
    }

    /// Merges generators with identical exponent columns, moves constant
    /// monomials into the center, and drops zero generators and unused factors.
    pub fn compact(&self) -> Self {
        let n = self.dim();
        let mut c = self.c.clone();
        let mut columns: Vec<(Vec<u32>, DVector<f64>)> = Vec::new();
        let mut index: HashMap<Vec<u32>, usize> = HashMap::new();
        for i in 0..self.g.ncols() {
            let exp: Vec<u32> = self.e.column(i).iter().copied().collect();
            let gv = self.g.column(i).into_owned();
            if exp.iter().all(|&x| x == 0) {
                c += gv;
            } else if let Some(&j) = index.get(&exp) {
                columns[j].1 += gv;
            } else {
                index.insert(exp.clone(), columns.len());
                columns.push((exp, gv));
            }
        }
        columns.retain(|(_, g)| g.iter().any(|&x| x != 0.0));
        let used: Vec<usize> = (0..self.id.len())
            .filter(|&k| columns.iter().any(|(e, _)| e[k] != 0))
            .collect();
        let gi_cols: Vec<_> = self
            .gi
            .column_iter()
            .filter(|col| col.iter().any(|&x| x != 0.0))
            .collect();
        Self {
            c,
            g: DMatrix::from_fn(n, columns.len(), |i, j| columns[j].1[i]),
            gi: DMatrix::from_fn(n, gi_cols.len(), |i, j| gi_cols[j][i]),
            e: DMatrix::from_fn(used.len(), columns.len(), |k, j| columns[j].0[used[k]]),
            id: used.iter().map(|&k| self.id[k]).collect(),
        }
    }

    fn monomial_range(&self, col: usize) -> MonomialRange {
        let exps = self.e.column(col);
        if exps.iter().all(|&x| x == 0) {
            MonomialRange::Constant
        } else if exps.iter().all(|&x| x % 2 == 0) {
            MonomialRange::UnitRange
        } else {
            MonomialRange::Symmetric
        }
    }

    /// Interval enclosure (treats every monomial as independent, so it is
    /// sound but can be loose; tighten with [`split`](Self::split)).
    pub fn interval_hull(&self) -> IntervalBox {
        let mut iv: Vec<Interval> = self.c.iter().map(|&x| Interval::point(x)).collect();
        for j in 0..self.g.ncols() {
            let range = self.monomial_range(j);
            for (i, &g) in self.g.column(j).iter().enumerate() {
                let (lo, hi) = match range {
                    MonomialRange::Constant => (g, g),
                    MonomialRange::UnitRange => (g.min(0.0), g.max(0.0)),
                    MonomialRange::Symmetric => (-g.abs(), g.abs()),
                };
                iv[i] = iv[i] + Interval::new(lo, hi);
            }
        }
        for (i, row) in self.gi.row_iter().enumerate() {
            let r: f64 = row.iter().map(|x| x.abs()).sum();
            iv[i] = iv[i] + Interval::new(-r, r);
        }
        IntervalBox(iv)
    }

    /// Zonotope enclosure: monomials with only even exponents range over
    /// `[0, 1]`, so they contribute `g/2 + (g/2)·β`.
    pub fn to_zonotope(&self) -> Zonotope {
        let mut c = self.c.clone();
        let mut gens = Vec::with_capacity(self.g.ncols() + self.gi.ncols());
        for j in 0..self.g.ncols() {
            let g = self.g.column(j).into_owned();
            match self.monomial_range(j) {
                MonomialRange::Constant => c += g,
                MonomialRange::UnitRange => {
                    c += &g * 0.5;
                    gens.push(g * 0.5);
                }
                MonomialRange::Symmetric => gens.push(g),
            }
        }
        gens.extend(self.gi.column_iter().map(|col| col.into_owned()));
        let n = self.dim();
        Zonotope::new(c, DMatrix::from_fn(n, gens.len(), |i, j| gens[j][i]))
    }

    /// Splits the domain of `factor` into `[-1, 0]` and `[0, 1]`. Each half
    /// is exact and uses a fresh id for the rescaled factor
    /// (`α = (α' ∓ 1) / 2`). Returns `(lower, upper)`.
    pub fn split(&self, factor: FactorId) -> (Self, Self) {
        let k = self
            .id
            .iter()
            .position(|&x| x == factor)
            .expect("factor not in this set");
        let half = |sign: f64| {
            let mut gens: Vec<DVector<f64>> = Vec::new();
            let mut exps: Vec<Vec<u32>> = Vec::new();
            for i in 0..self.g.ncols() {
                let d = self.e[(k, i)];
                // (α'/2 + sign/2)^d = Σ_j C(d, j) (1/2)^d α'^j sign^(d-j)
                for j in 0..=d {
                    let coef = binomial(d, j) * 0.5f64.powi(d as i32) * sign.powi((d - j) as i32);
                    let mut exp: Vec<u32> = self.e.column(i).iter().copied().collect();
                    exp[k] = j;
                    exps.push(exp);
                    gens.push(self.g.column(i) * coef);
                }
            }
            let mut id = self.id.clone();
            id[k] = FactorId::fresh();
            let (n, p) = (self.dim(), id.len());
            Self {
                c: self.c.clone(),
                g: DMatrix::from_fn(n, gens.len(), |r, c| gens[c][r]),
                gi: self.gi.clone(),
                e: DMatrix::from_fn(p, exps.len(), |r, c| exps[c][r]),
                id,
            }
            .compact()
        };
        (half(-1.0), half(1.0))
    }
}

fn binomial(n: u32, k: u32) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
}

/// Union of the two id lists, with both exponent matrices re-indexed to it.
fn merge_ids(
    a: &SparsePolyZonotope,
    b: &SparsePolyZonotope,
) -> (Vec<FactorId>, DMatrix<u32>, DMatrix<u32>) {
    let mut id = a.id.clone();
    let mut pos: HashMap<FactorId, usize> = id.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    for &k in &b.id {
        pos.entry(k).or_insert_with(|| {
            id.push(k);
            id.len() - 1
        });
    }
    let p = id.len();
    let e1 = DMatrix::from_fn(p, a.e.ncols(), |r, c| {
        if r < a.e.nrows() { a.e[(r, c)] } else { 0 }
    });
    let mut e2 = DMatrix::zeros(p, b.e.ncols());
    for (r, k) in b.id.iter().enumerate() {
        for c in 0..b.e.ncols() {
            e2[(pos[k], c)] = b.e[(r, c)];
        }
    }
    (id, e1, e2)
}

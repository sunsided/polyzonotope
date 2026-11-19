//! Sparse polynomial zonotopes (Kochdumper & Althoff, IEEE TAC 2021).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;
use portable_atomic::{AtomicU64, Ordering};

use nalgebra::{DMatrix, DVector};

use crate::round::{self, RowErrors};
use crate::{Interval, IntervalBox, Zonotope, from_column_list, hcat};

/// Global identifier of a dependent factor `α_k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FactorId(u64);

/// Next unused [`FactorId`] value; shared process-wide so ids never collide.
///
/// `portable_atomic` rather than `core`, so targets without native 64-bit
/// atomics (Cortex-M, 32-bit RISC-V) still get a shared counter.
static NEXT_FACTOR: AtomicU64 = AtomicU64::new(0);

impl FactorId {
    /// A new id, distinct from every id handed out before in this process.
    #[must_use]
    pub fn fresh() -> Self {
        Self(NEXT_FACTOR.fetch_add(1, Ordering::Relaxed))
    }

    /// `n` new ids, distinct from each other and from every earlier id.
    #[must_use]
    pub fn fresh_n(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::fresh()).collect()
    }
}

/// Why the parts passed to [`SparsePolyZonotope::new`] do not fit together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// The dependent generator matrix `G` does not have one row per dimension.
    GeneratorRows,
    /// The independent generator matrix `GI` does not have one row per dimension.
    IndependentRows,
    /// The exponent matrix `E` does not have one column per dependent generator.
    ExponentColumns,
    /// The exponent matrix `E` does not have one row per factor id.
    ExponentRows,
    /// The same [`FactorId`] appears more than once.
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

impl core::error::Error for ShapeError {}

/// Sparse polynomial zonotope `⟨c, G, GI, E, id⟩`.
#[derive(Clone, Debug, PartialEq)]
///
/// Invariants (checked by [`new`](Self::new)): `g` and `gi` have one row per
/// entry of `c`, `e` has one column per column of `g` and one row per id, and
/// the ids are distinct.
pub struct SparsePolyZonotope {
    /// Center `c`; its length is the dimension of the set.
    c: DVector<f64>,
    /// Dependent generators `G`, one per column; column `i` is scaled by monomial `i`.
    g: DMatrix<f64>,
    /// Independent generators `GI`, one per column, each scaled by its own `β_j`.
    gi: DMatrix<f64>,
    /// Exponents `E`: entry `(k, i)` is the power of `α_k` in monomial `i`.
    e: DMatrix<u32>,
    /// Global ids of the dependent factors; row `k` of `e` belongs to `id[k]`.
    id: Vec<FactorId>,
}

/// How a dependent monomial ranges over `α ∈ [-1, 1]^p`.
enum MonomialRange {
    /// All exponents are zero: the monomial is exactly 1.
    Constant,
    /// All exponents are even: the monomial ranges over `[0, 1]`.
    UnitRange,
    /// Some exponent is odd: the monomial ranges over `[-1, 1]`.
    Symmetric,
}

impl MonomialRange {
    /// The range of `g · m` as the monomial `m` ranges over this range.
    fn scaled_bounds(&self, g: f64) -> Interval {
        match self {
            Self::Constant => Interval::point(g),
            Self::UnitRange => Interval::new(g.min(0.0), g.max(0.0)),
            Self::Symmetric => Interval::new(-g.abs(), g.abs()),
        }
    }
}

impl SparsePolyZonotope {
    /// The set `⟨c, g, gi, e, id⟩`, after checking that the shapes fit together.
    ///
    /// # Errors
    ///
    /// - [`ShapeError::GeneratorRows`] if `g` does not have `c.len()` rows.
    /// - [`ShapeError::IndependentRows`] if `gi` does not have `c.len()` rows.
    /// - [`ShapeError::ExponentColumns`] if `e` and `g` differ in column count.
    /// - [`ShapeError::ExponentRows`] if `e` does not have one row per id.
    /// - [`ShapeError::DuplicateId`] if an id appears twice.
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
    #[must_use]
    pub fn from_zonotope(z: &Zonotope) -> Self {
        let m = z.num_generators();
        Self {
            c: z.center().clone(),
            g: z.generators().clone(),
            gi: DMatrix::zeros(z.dim(), 0),
            e: DMatrix::identity(m, m),
            id: FactorId::fresh_n(m),
        }
    }

    /// Dimension of the space the set lives in.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.c.len()
    }

    /// Center `c`.
    #[must_use]
    pub fn center(&self) -> &DVector<f64> {
        &self.c
    }

    /// Dependent generators `G`, one per column, matching the columns of [`exponents`](Self::exponents).
    #[must_use]
    pub fn dependent_generators(&self) -> &DMatrix<f64> {
        &self.g
    }

    /// Independent generators `GI`, one per column.
    #[must_use]
    pub fn independent_generators(&self) -> &DMatrix<f64> {
        &self.gi
    }

    /// Exponent matrix `E`: entry `(k, i)` is the power of the factor `ids()[k]` in monomial `i`.
    #[must_use]
    pub fn exponents(&self) -> &DMatrix<u32> {
        &self.e
    }

    /// Ids of the dependent factors, one per row of [`exponents`](Self::exponents).
    #[must_use]
    pub fn ids(&self) -> &[FactorId] {
        &self.id
    }

    /// Point of the set for the given factor values (`alpha` looked up by
    /// id, `beta` positional).
    ///
    /// # Panics
    ///
    /// Panics if `beta` does not have one entry per independent generator.
    #[must_use]
    pub fn evaluate_with(&self, alpha: impl Fn(FactorId) -> f64, beta: &[f64]) -> DVector<f64> {
        assert_eq!(
            beta.len(),
            self.gi.ncols(),
            "one β per independent generator"
        );

        let a: Vec<f64> = self.id.iter().map(|&k| alpha(k)).collect();

        self.evaluate_positional(&a, beta)
    }

    /// Point of the set for positional factor values: `a[k]` belongs to `ids()[k]`.
    ///
    /// Callers check that `a` has one entry per id and `beta` one entry per
    /// independent generator.
    fn evaluate_positional(&self, a: &[f64], beta: &[f64]) -> DVector<f64> {
        let mut p = &self.c + &self.gi * DVector::from_column_slice(beta);
        for i in 0..self.g.ncols() {
            p += self.g.column(i) * self.monomial_value(i, a);
        }

        p
    }

    /// Value of monomial `i`, `Π_k a[k]^E[k, i]`, for positional factor values `a`.
    fn monomial_value(&self, i: usize, a: &[f64]) -> f64 {
        a.iter()
            .zip(self.e.column(i).iter())
            .map(|(&x, &d)| powu(x, d))
            .product()
    }

    /// Positional variant of [`evaluate_with`](Self::evaluate_with): `alpha[k]` belongs to `ids()[k]`.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` does not have one entry per id, or `beta` one entry
    /// per independent generator.
    #[must_use]
    pub fn evaluate(&self, alpha: &[f64], beta: &[f64]) -> DVector<f64> {
        assert_eq!(alpha.len(), self.id.len(), "one α per factor");
        assert_eq!(
            beta.len(),
            self.gi.ncols(),
            "one β per independent generator"
        );

        self.evaluate_positional(alpha, beta)
    }

    /// Exact: `{ M x | x ∈ PZ }`.
    ///
    /// # Panics
    ///
    /// Panics if `m` does not have one column per dimension.
    #[must_use]
    pub fn linear_map(&self, m: &DMatrix<f64>) -> Self {
        let mut errs = RowErrors::new(m.nrows());
        let c = round::mat_vec(m, &self.c, &mut errs);
        let g = round::mat_mul(m, &self.g, &mut errs);
        let gi = round::mat_mul(m, &self.gi, &mut errs);

        Self {
            c,
            g,
            gi: hcat(&gi, &errs.generators()),
            e: self.e.clone(),
            id: self.id.clone(),
        }
    }

    /// Exact: `{ x + v | x ∈ PZ }`.
    ///
    /// # Panics
    ///
    /// Panics if `v` does not have the set's dimension.
    #[must_use]
    pub fn translate(&self, v: &DVector<f64>) -> Self {
        let mut errs = RowErrors::new(self.dim());
        let c = round::vec_add(&self.c, v, &mut errs);

        Self {
            c,
            g: self.g.clone(),
            gi: hcat(&self.gi, &errs.generators()),
            e: self.e.clone(),
            id: self.id.clone(),
        }
    }

    /// Same set, with all dependent factors renamed to fresh ids (decorrelates it from every other set).
    #[must_use]
    pub fn with_fresh_ids(&self) -> Self {
        Self {
            id: FactorId::fresh_n(self.id.len()),
            ..self.clone()
        }
    }

    /// Exact Minkowski sum `{ x + y | x ∈ self, y ∈ other }` (sets treated as independent).
    ///
    /// # Panics
    ///
    /// Panics if the two sets have different dimensions.
    #[must_use]
    pub fn minkowski_sum(&self, other: &Self) -> Self {
        self.exact_plus(&other.with_fresh_ids())
    }

    /// Exact addition respecting shared factors: `{ p₁(α) + p₂(α) }`.
    ///
    /// # Panics
    ///
    /// Panics if the two sets have different dimensions.
    #[must_use]
    pub fn exact_plus(&self, other: &Self) -> Self {
        let (id, e1, e2) = merge_ids(self, other);
        let mut errs = RowErrors::new(self.dim());
        let c = round::vec_add(&self.c, &other.c, &mut errs);
        Self {
            c,
            g: hcat(&self.g, &other.g),
            gi: hcat(&hcat(&self.gi, &other.gi), &errs.generators()),
            e: hcat(&e1, &e2),
            id,
        }
        .compact()
    }

    /// Exact Cartesian product, keeping shared factors correlated.
    ///
    /// The result is [`compact`](Self::compact)ed, like [`exact_plus`](Self::exact_plus).
    #[must_use]
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
        .compact()
    }

    /// Merges generators with identical exponent columns, moves constant
    /// monomials into the center, and drops zero dependent generators and
    /// unused factors.
    ///
    /// Independent generators are kept as they are, even zero ones, so the
    /// positions of the `β` values stay stable across [`exact_plus`](Self::exact_plus)
    /// and [`split`](Self::split). The price is that zero independent
    /// generators, e.g. from a singular [`linear_map`](Self::linear_map), are
    /// never reclaimed here; [`to_zonotope`](Self::to_zonotope) leaves them out.
    /// With the `sound` feature, bounds on the rounding error of the merged
    /// sums are appended after them.
    #[must_use]
    pub fn compact(&self) -> Self {
        let n = self.dim();
        let mut c = self.c.clone();
        let mut errs = RowErrors::new(n);
        let mut columns: Vec<(Vec<u32>, DVector<f64>)> = Vec::new();
        let mut index: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
        for i in 0..self.g.ncols() {
            let exp: Vec<u32> = self.e.column(i).iter().copied().collect();
            let gv = self.g.column(i).into_owned();
            if exp.iter().all(|&x| x == 0) {
                round::add_into(&mut c, &gv, &mut errs);
            } else if let Some(&j) = index.get(&exp) {
                round::add_into(&mut columns[j].1, &gv, &mut errs);
            } else {
                index.insert(exp.clone(), columns.len());
                columns.push((exp, gv));
            }
        }

        columns.retain(|(_, g)| g.iter().any(|&x| x != 0.0));
        let used: Vec<usize> = (0..self.id.len())
            .filter(|&k| columns.iter().any(|(e, _)| e[k] != 0))
            .collect();
        Self {
            c,
            g: DMatrix::from_fn(n, columns.len(), |i, j| columns[j].1[i]),
            gi: hcat(&self.gi, &errs.generators()),
            e: DMatrix::from_fn(used.len(), columns.len(), |k, j| columns[j].0[used[k]]),
            id: used.iter().map(|&k| self.id[k]).collect(),
        }
    }

    /// How monomial `col` ranges over the factor domain, from the parity of its exponents.
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
    /// sound but can be loose; tighten with [`split`](Self::split)). With the
    /// `sound` feature the bounds are rounded outward.
    #[must_use]
    pub fn interval_hull(&self) -> IntervalBox {
        let mut iv: Vec<Interval> = self.c.iter().map(|&x| Interval::point(x)).collect();
        for j in 0..self.g.ncols() {
            let range = self.monomial_range(j);
            for (i, &g) in self.g.column(j).iter().enumerate() {
                iv[i] = iv[i] + range.scaled_bounds(g);
            }
        }

        for (i, row) in self.gi.row_iter().enumerate() {
            let r = round::abs_sum_up(row.iter());
            iv[i] = iv[i] + Interval::new(-r, r);
        }

        IntervalBox(iv)
    }

    /// Zonotope enclosure: monomials with only even exponents range over
    /// `[0, 1]`, so they contribute `g/2 + (g/2)·β`.
    ///
    /// All-zero generators, dependent or independent, are left out, so
    /// [`Zonotope::order`] counts only generators that shape the set.
    #[must_use]
    pub fn to_zonotope(&self) -> Zonotope {
        let n = self.dim();
        let mut c = self.c.clone();
        let mut errs = RowErrors::new(n);
        let mut gens = Vec::with_capacity(self.g.ncols() + self.gi.ncols());
        for j in 0..self.g.ncols() {
            let g = self.g.column(j).into_owned();
            match self.monomial_range(j) {
                MonomialRange::Constant => round::add_into(&mut c, &g, &mut errs),
                MonomialRange::UnitRange => {
                    let half = round::scaled(&g, 0.5, &mut errs);
                    round::add_into(&mut c, &half, &mut errs);
                    gens.push(half);
                }
                MonomialRange::Symmetric => gens.push(g),
            }
        }

        gens.extend(self.gi.column_iter().map(nalgebra::Matrix::into_owned));
        gens.retain(|g| g.iter().any(|&x| x != 0.0));
        Zonotope::new(c, hcat(&from_column_list(n, &gens), &errs.generators()))
    }

    /// Splits the domain of `factor` into `[-1, 0]` and `[0, 1]`. Each half
    /// is exact and uses a fresh id for the rescaled factor
    /// (`α = (α' ∓ 1) / 2`). Returns `(lower, upper)`.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not one of [`ids`](Self::ids), or if it appears
    /// with an exponent above 56, where the binomial coefficients of the
    /// rescaling are no longer exact in `f64`.
    #[must_use]
    pub fn split(&self, factor: FactorId) -> (Self, Self) {
        let k = self
            .id
            .iter()
            .position(|&x| x == factor)
            .expect("factor not in this set");

        (self.half_domain(k, -1.0), self.half_domain(k, 1.0))
    }

    /// The half of the set where factor `k` lies between 0 and `sign`, with
    /// `α_k = (α' + sign) / 2` substituted for a fresh factor `α'`.
    fn half_domain(&self, k: usize, sign: f64) -> Self {
        let mut gens: Vec<DVector<f64>> = Vec::new();
        let mut exps: Vec<Vec<u32>> = Vec::new();
        let mut errs = RowErrors::new(self.dim());
        for i in 0..self.g.ncols() {
            let column = self.g.column(i).into_owned();
            for (exp, coef) in self.expand_rescaled(i, k, sign) {
                exps.push(exp);
                gens.push(round::scaled(&column, coef, &mut errs));
            }
        }

        let mut id = self.id.clone();
        id[k] = FactorId::fresh();
        let (n, p) = (self.dim(), id.len());

        Self {
            c: self.c.clone(),
            g: from_column_list(n, &gens),
            gi: hcat(&self.gi, &errs.generators()),
            e: from_column_list(p, &exps),
            id,
        }
        .compact()
    }

    /// Binomial expansion of monomial `i` after substituting
    /// `α_k = (α' + sign) / 2`: one `(exponents, coefficient)` pair per power
    /// `j` of `α'`, from
    /// `(α'/2 + sign/2)^d = Σ_j C(d, j) (1/2)^d α'^j sign^(d-j)`.
    fn expand_rescaled(&self, i: usize, k: usize, sign: f64) -> Vec<(Vec<u32>, f64)> {
        let d = self.e[(k, i)];
        assert!(
            d <= MAX_SPLIT_EXPONENT,
            "split needs exponents of at most {MAX_SPLIT_EXPONENT}, found {d}"
        );

        // Repeated squaring, not `powi`: powers of 1/2 and ±1 are exact that way.
        (0..=d)
            .map(|j| {
                let coef = binomial(d, j) * pow_by_squaring(0.5, d) * pow_by_squaring(sign, d - j);
                let mut exp: Vec<u32> = self.e.column(i).iter().copied().collect();
                exp[k] = j;
                (exp, coef)
            })
            .collect()
    }
}

/// `x^n`, via [`f64::powi`] with the `std` feature and repeated squaring without it.
///
/// Exponents beyond `i32::MAX` take the repeated-squaring path in both cases.
fn powu(x: f64, n: u32) -> f64 {
    #[cfg(feature = "std")]
    if let Ok(n) = i32::try_from(n) {
        return x.powi(n);
    }

    pow_by_squaring(x, n)
}

/// `x^n` by repeated squaring, in `O(log n)` multiplications; `x^0` is 1, also for `x = 0`.
fn pow_by_squaring(mut x: f64, mut n: u32) -> f64 {
    let mut acc = 1.0;
    while n > 0 {
        if n & 1 == 1 {
            acc *= x;
        }

        x *= x;
        n >>= 1;
    }

    acc
}

/// Largest exponent [`SparsePolyZonotope::split`] accepts: every `C(d, j)` with
/// `d <= 56` stays below `2^53`, so [`binomial`] and the coefficients
/// `C(d, j) / 2^d` of the rescaling are exact.
const MAX_SPLIT_EXPONENT: u32 = 56;

/// Binomial coefficient `C(n, k)` for `k <= n`, exact for `n <=` [`MAX_SPLIT_EXPONENT`].
///
/// Integer arithmetic keeps every step exact: `C(n, i) (n - i)` is divisible by
/// `i + 1` and below `2^63` for those `n`.
#[expect(
    clippy::cast_precision_loss,
    reason = "C(n, k) < 2^53 for n <= MAX_SPLIT_EXPONENT, so the conversion is exact"
)]
fn binomial(n: u32, k: u32) -> f64 {
    (0..k).fold(1u64, |acc, i| acc * u64::from(n - i) / u64::from(i + 1)) as f64
}

/// Union of the two id lists, with both exponent matrices re-indexed to it.
fn merge_ids(
    a: &SparsePolyZonotope,
    b: &SparsePolyZonotope,
) -> (Vec<FactorId>, DMatrix<u32>, DMatrix<u32>) {
    let mut id = a.id.clone();
    let mut pos: BTreeMap<FactorId, usize> = id.iter().enumerate().map(|(i, &k)| (k, i)).collect();
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

/// Unit tests for the private power helpers, which the integration tests only
/// reach through the `std` path.
#[cfg(test)]
mod tests {
    use super::{pow_by_squaring, powu};

    /// `x^n` as `n` plain multiplications, the reference both helpers must match.
    fn naive_pow(x: f64, n: u32) -> f64 {
        (0..n).fold(1.0, |acc, _| acc * x)
    }

    /// Repeated squaring agrees with plain multiplication, including `x^0 = 1` for `x = 0`.
    #[test]
    fn pow_by_squaring_matches_repeated_multiplication() {
        for x in [0.0, 1.0, -1.0, 0.5, -1.5, 0.3] {
            for n in 0..=40 {
                let (got, want) = (pow_by_squaring(x, n), naive_pow(x, n));
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs(),
                    "{x}^{n}: {got} vs {want}"
                );
            }
        }
    }

    /// Exponents beyond `i32::MAX` take the repeated-squaring path instead of panicking.
    #[test]
    fn powu_handles_exponents_beyond_i32_max() {
        assert!((powu(1.0, u32::MAX) - 1.0).abs() < f64::EPSILON);
        assert!((powu(-1.0, u32::MAX) + 1.0).abs() < f64::EPSILON);
        assert!(powu(0.5, u32::MAX).abs() < f64::EPSILON);
    }
}

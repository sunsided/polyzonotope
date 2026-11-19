use nalgebra::DVector;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "empty interval [{lo}, {hi}]");
        Self { lo, hi }
    }
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
    pub fn contains(&self, x: f64, tol: f64) -> bool {
        self.lo - tol <= x && x <= self.hi + tol
    }
}

impl std::ops::Add for Interval {
    type Output = Interval;
    fn add(self, o: Interval) -> Interval {
        Interval {
            lo: self.lo + o.lo,
            hi: self.hi + o.hi,
        }
    }
}

/// Axis-aligned box: one interval per dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalBox(pub Vec<Interval>);

impl IntervalBox {
    pub fn dim(&self) -> usize {
        self.0.len()
    }
    pub fn contains(&self, p: &DVector<f64>, tol: f64) -> bool {
        p.len() == self.dim()
            && self
                .0
                .iter()
                .zip(p.iter())
                .all(|(iv, &x)| iv.contains(x, tol))
    }
}

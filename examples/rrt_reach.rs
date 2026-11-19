//! Rapidly-exploring random tree that keeps an edge only if the reachable set
//! along it provably misses every obstacle.
//!
//! The robot is a point that follows each edge in steps of `DT`, with a
//! feedback controller that shrinks its tracking error by `CONTRACTION` per
//! step while a bounded disturbance adds up to `DISTURBANCE` per axis. The
//! error set therefore evolves as `E' = CONTRACTION · E ⊕ W`, a zonotope. Each
//! tree node stores the error set it is reached with, so uncertainty builds up
//! along a branch, and `reduce` keeps the representation small.
//!
//! Between steps the error is assumed to stay in the error sets before or
//! after the step, so the robot is covered by the box around both, swept along
//! the reference segment.
//!
//! An edge is kept only if every step's swept set is `Disjoint` from every
//! obstacle. `Intersects` and `Unknown` both reject it: only the proof counts.
//! (The proof is rigorous only with the `sound` feature; this example runs
//! in plain `f64`.)
//!
//! Run: `cargo run --example rrt_reach`, then open `target/plots/rrt.svg`.

use nalgebra::{DMatrix, DVector};
use plotters::prelude::*;
use polyzonotope::{Intersection, Interval, IntervalBox, Polytope, Reduction, Zonotope};

/// Largest distance one edge covers.
const STEP: f64 = 0.8;
/// Distance the reference moves per control step.
const DT: f64 = 0.2;
/// Factor by which feedback shrinks the tracking error per control step.
const CONTRACTION: f64 = 0.85;
/// Half width of the disturbance box added per control step.
const DISTURBANCE: f64 = 0.03;
/// Order the error sets are reduced to after each step.
const ORDER: f64 = 4.0;
/// Split budget for each obstacle check.
const SPLITS: usize = 4;

/// A tree node: a position and the tracking-error set it is reached with.
struct Node {
    /// Reference position.
    at: DVector<f64>,
    /// Tracking error around `at`, a zonotope centered near the origin.
    error: Zonotope,
    /// Index of the parent node; `None` for the root.
    parent: Option<usize>,
}

/// A deterministic xorshift generator, enough for sampling the plane.
struct Rng(u64);

impl Rng {
    /// The next value in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // The top 53 bits as a fraction.
        f64::from(u32::try_from(self.0 >> 43).expect("21 bits fit")) / f64::from(1u32 << 21)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let obstacles: Vec<IntervalBox> = [
        (3.0, 4.0, 0.0, 6.5),
        (6.0, 7.0, 3.5, 10.0),
        (1.0, 2.5, 7.0, 8.0),
        (7.5, 9.0, 1.5, 2.5),
    ]
    .iter()
    .map(|&(x0, x1, y0, y1)| IntervalBox(vec![Interval::new(x0, x1), Interval::new(y0, y1)]))
    .collect();
    let polytopes: Vec<Polytope> = obstacles.iter().map(Polytope::from_box).collect();
    let (start, goal) = (
        DVector::from_vec(vec![1.0, 1.0]),
        DVector::from_vec(vec![9.0, 9.0]),
    );

    let initial_error = Zonotope::from_box(&IntervalBox(vec![Interval::new(-0.1, 0.1); 2]));
    let mut tree = vec![Node {
        at: start,
        error: initial_error,
        parent: None,
    }];
    let mut rejected = Vec::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut reached = None;
    for _ in 0..3000 {
        let sample = if rng.next_unit() < 0.1 {
            goal.clone()
        } else {
            DVector::from_vec(vec![10.0 * rng.next_unit(), 10.0 * rng.next_unit()])
        };
        let near = nearest(&tree, &sample);
        let to = steer(&tree[near].at, &sample);
        match follow_edge(&tree[near], &to, &polytopes) {
            Some(error) => {
                tree.push(Node {
                    at: to.clone(),
                    error,
                    parent: Some(near),
                });
                if (&to - &goal).norm() < 0.3 {
                    reached = Some(tree.len() - 1);
                    break;
                }
            }
            None => rejected.push((tree[near].at.clone(), to)),
        }
    }

    println!(
        "{} nodes, {} rejected edges, goal reached: {}",
        tree.len(),
        rejected.len(),
        reached.is_some()
    );
    plot(&tree, &rejected, &obstacles, reached)
}

/// Index of the tree node closest to `p`.
fn nearest(tree: &[Node], p: &DVector<f64>) -> usize {
    (0..tree.len())
        .min_by(|&a, &b| {
            (&tree[a].at - p)
                .norm()
                .total_cmp(&(&tree[b].at - p).norm())
        })
        .expect("the tree has a root")
}

/// The point at most [`STEP`] from `from` in the direction of `to`.
fn steer(from: &DVector<f64>, to: &DVector<f64>) -> DVector<f64> {
    let d = to - from;
    let len = d.norm();
    if len <= STEP {
        to.clone()
    } else {
        from + d * (STEP / len)
    }
}

/// Error set at the end of the edge from `node` to `to`, or `None` if some
/// step's swept set is not proven to miss every obstacle.
fn follow_edge(node: &Node, to: &DVector<f64>, obstacles: &[Polytope]) -> Option<Zonotope> {
    let steps = ((to - &node.at).norm() / DT).ceil().max(1.0);
    let delta = (to - &node.at) / steps;
    let contraction = DMatrix::from_diagonal_element(2, 2, CONTRACTION);
    let disturbance =
        Zonotope::from_box(&IntervalBox(vec![
            Interval::new(-DISTURBANCE, DISTURBANCE);
            2
        ]));
    // The reference moving by `delta` during one step, as a segment around its midpoint.
    let segment = Zonotope::new(
        DVector::zeros(2),
        DMatrix::from_column_slice(2, 1, (&delta / 2.0).as_slice()),
    );

    let mut error = node.error.clone();
    let mut at = node.at.clone();
    let mut taken = 0.0;
    while taken < steps {
        taken += 1.0;
        let midpoint = &at + &delta / 2.0;
        // During the step the error is in `error` or `next` (an assumption of this
        // discrete model); the box around both, swept along the segment, covers the robot.
        let next = error
            .linear_map(&contraction)
            .minkowski_sum(&disturbance)
            .reduce(ORDER, Reduction::Girard);
        let swept = covering_box(&error, &next)
            .minkowski_sum(&segment)
            .translate(&midpoint);
        if obstacles
            .iter()
            .any(|o| swept.intersection(o, SPLITS) != Intersection::Disjoint)
        {
            return None;
        }

        error = next;
        at += &delta;
    }

    Some(error)
}

/// The axis-aligned box around both sets, as a zonotope.
fn covering_box(a: &Zonotope, b: &Zonotope) -> Zonotope {
    let (ha, hb) = (a.interval_hull(), b.interval_hull());
    let union =
        (0..2).map(|i| Interval::new(ha.0[i].lo.min(hb.0[i].lo), ha.0[i].hi.max(hb.0[i].hi)));

    Zonotope::from_box(&IntervalBox(union.collect()))
}

/// Draws obstacles, rejected edges, the tree and the reachable sets along the path found.
fn plot(
    tree: &[Node],
    rejected: &[(DVector<f64>, DVector<f64>)],
    obstacles: &[IntervalBox],
    reached: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("target/plots")?;
    let root = SVGBackend::new("target/plots/rrt.svg", (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(0f64..10f64, 0f64..10f64)?;

    chart.draw_series(obstacles.iter().map(|b| {
        Rectangle::new(
            [(b.0[0].lo, b.0[1].lo), (b.0[0].hi, b.0[1].hi)],
            BLACK.mix(0.6).filled(),
        )
    }))?;
    chart.draw_series(
        rejected
            .iter()
            .map(|(a, b)| PathElement::new(vec![(a[0], a[1]), (b[0], b[1])], RED.mix(0.25))),
    )?;
    chart.draw_series(tree.iter().filter_map(|n| {
        let p = &tree[n.parent?].at;
        Some(PathElement::new(
            vec![(p[0], p[1]), (n.at[0], n.at[1])],
            BLUE.mix(0.4),
        ))
    }))?;

    let mut path = Vec::new();
    let mut cursor = reached;
    while let Some(i) = cursor {
        path.push(i);
        cursor = tree[i].parent;
    }
    chart.draw_series(path.iter().map(|&i| {
        let set = tree[i].error.translate(&tree[i].at);
        Polygon::new(
            set.polygon()
                .into_iter()
                .map(|[x, y]| (x, y))
                .collect::<Vec<_>>(),
            GREEN.mix(0.4),
        )
    }))?;

    root.present()?;
    println!("wrote target/plots/rrt.svg");
    Ok(())
}

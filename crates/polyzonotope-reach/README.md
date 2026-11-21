# polyzonotope-reach

Reachable-set over-approximation for dynamical systems, with the sparse polynomial zonotopes of [`polyzonotope`](https://crates.io/crates/polyzonotope). Not released yet.

## Linear systems

`LinearReach` propagates `x' = A x + B u`, where the input `u(t)` can be any integrable signal with values in a zonotope `U`. Each step `[t_k, t_k + r]` returns two sets:

- `time_point` holds every state at `t_k + r`: `e^{Ar} X ⊕ Γ(r) B c_u ⊕ R_p(B U_Δ)`.
- `time_interval` holds every state at any `t` in the step. It is the hull of `X` and its successor (Girard 2005), plus the curvature corrections `F X` and `F̃ B c_u`, plus `R_p`.

The matrix exponential is a truncated Taylor series with its remainder `E(r)` carried as an interval matrix (Althoff 2010). `c_u` is the center of `U`, and `R_p` bounds the response to the varying part `U_Δ`.

```rust
use nalgebra::{DMatrix, DVector};
use polyzonotope::{SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{LinearReach, LinearSystem};

// A damped oscillator pushed by an input in [0.9, 1.1].
let system = LinearSystem {
    a: DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, -0.2]),
    b: DMatrix::from_row_slice(2, 1, &[0.0, 1.0]),
};
let reach = LinearReach::new(system, 0.1, 6)?;
let start = SparsePolyZonotope::from_zonotope(&Zonotope::new(
    DVector::from_vec(vec![1.0, 0.0]),
    DMatrix::from_diagonal_element(2, 2, 0.05),
));
let input = Zonotope::new(DVector::from_vec(vec![1.0]), DMatrix::from_element(1, 1, 0.1));

let step = reach.step(&start, &input);
let hull = step.time_interval.interval_hull();
assert!(hull.0[0].lo <= 0.95 && hull.0[0].hi >= 1.05);
# Ok::<(), polyzonotope_reach::ReachError>(())
```

Sets grow by a few generators per step. Call `reduce` from `polyzonotope` between steps to keep them small.

## Soundness

The over-approximation is sound in real arithmetic. The Taylor remainder, the time-interval corrections and the input response are all enclosed, so the sets are larger than the true reachable set, never smaller.

With the `sound` feature, the rounding of the floating-point computation is covered as well. The series terms, the factorials and the remainder are computed in outward-rounded interval arithmetic, and every set operation tracks its own rounding errors. The one exception is the coefficient `i^{-i/(i-1)} − i^{-1/(i-1)}` of the time-interval correction. It comes from `libm::pow`, which is not correctly rounded, so it is widened by a relative `1e-12` instead.

## Verification

The tests compare against reference trajectories: matrix exponentials of the augmented system, and inputs that jump inside `U`. They check the state at the end of each step and at sampled times inside it, against both the interval hull and the support function in random directions. Without inputs, the time-point set must equal `e^{Ar} X` at matching factor values. Removing the Taylor remainder, `F`, `F̃` or `R_p` makes at least one test fail.

## Roadmap

1. Nonlinear systems by conservative polynomialization (Althoff 2013; Kochdumper & Althoff 2021, Alg. 1): a second-order Taylor model via `quadratic_map`, a Lagrange remainder from a user-supplied bound on the third derivatives, and the iterative abstraction-error check.
2. A cross-check against CORA's `linearSys` reach (it runs under Octave), and a generator script for the nonlinear case, which needs MATLAB because CORA's symbolic derivatives do not run under Octave.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/sunsided/polyzonotope/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/sunsided/polyzonotope/blob/main/LICENSE-MIT) at your option.

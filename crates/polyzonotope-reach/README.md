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

## Nonlinear systems

`NonlinearReach` propagates `x' = f(x, u)` for any `Dynamics`: a type that encloses `f`, its Jacobian and its Hessians over boxes in interval arithmetic. `Polynomial` implements it for polynomial vector fields and derives the Jacobian and Hessians itself. Each step does conservative linearization (Althoff, Stursberg & Buss 2008):

1. Linearize at `z* = (x*, c_u)`, where `x*` is the middle of the set moved half a step along `f`. This gives `x' = A x + B u + w`.
2. Run the linear step with the abstraction error as an extra input, bounded by a guessed box `Ψ̄`.
3. Bound the true error over the time-interval set `R(τ)` and `U`. By Taylor's theorem with the Lagrange remainder, it is `Δzᵀ Q Δz` plus a remainder. `Q` is the center of `½ H` over a box that holds every segment from `z*` into `R(τ) × U`. The quadratic part is a polynomial zonotope (`quadratic_map`), so its hull sees `x₀²` as non-negative. The spread of the Hessian and of the Jacobian go through intervals.
4. Accept the step if the error lies in `Ψ̄`. Otherwise widen `Ψ̄` and retry. After `max_iterations` tries, refuse with `ReachError::AbstractionDiverged`.

Both sets are reduced to the configured order after every step. `reach` runs many steps. A piece that diverges is enclosed in an oriented box, split along its longest axis and retried, up to `max_pieces` pieces per step.

```rust
use nalgebra::{DMatrix, DVector};
use polyzonotope::{SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{NonlinearReach, NonlinearSettings, Polynomial};

// Van der Pol: x₀' = x₁, x₁' = (1 − x₀²) x₁ − x₀.
let vdp = Polynomial::new(2, 0)
    .term(0, 1.0, &[0, 1])
    .term(1, 1.0, &[0, 1])
    .term(1, -1.0, &[2, 1])
    .term(1, -1.0, &[1, 0]);
let reach = NonlinearReach::new(vdp, NonlinearSettings::new(0.02));
let start = SparsePolyZonotope::from_zonotope(&Zonotope::new(
    DVector::from_vec(vec![1.4, 2.4]),
    DMatrix::from_diagonal_element(2, 2, 0.05),
));
let no_input = Zonotope::new(DVector::zeros(0), DMatrix::zeros(0, 0));

let flowpipe = reach.reach(&start, &no_input, 50)?;
assert_eq!(flowpipe.len(), 50);
# Ok::<(), polyzonotope_reach::ReachError>(())
```

A refusal means that no set was proven for that step. The crate does not hand back a set it could not confirm.

## Soundness

The over-approximation is sound in real arithmetic. The Taylor remainder, the time-interval corrections and the input response are all enclosed, so the sets are larger than the true reachable set, never smaller.

For nonlinear systems, the argument is the usual one for the error check. If the error over `R(τ)` lies in `Ψ̄`, no trajectory can leave `R(τ)` during the step. So the linear model plus an input in `Ψ̄` covers every trajectory.

With the `sound` feature, the rounding of the floating-point computation is covered as well. The series terms, the factorials and the remainder are computed in outward-rounded interval arithmetic, and every set operation tracks its own rounding errors. The nonlinear step evaluates `f`, the Jacobian and the Hessians in interval arithmetic. The model offset and the error radii are also computed that way, so they are rounded outward too. The one exception is the coefficient `i^{-i/(i-1)} − i^{-1/(i-1)}` of the time-interval correction. It comes from `libm::pow`, which is not correctly rounded, so it is widened by a relative `1e-12` instead.

## Verification

The linear tests compare against reference trajectories: matrix exponentials of the augmented system, and inputs that jump inside `U`. They check the state at the end of each step and at sampled times inside it, against both the interval hull and the support function in random directions. Without inputs, the time-point set must equal `e^{Ar} X` at matching factor values. Removing the Taylor remainder, `F`, `F̃` or `R_p` makes at least one test fail.

The nonlinear tests check four systems: Van der Pol with an input, a pendulum with a torque input, a cubic drift that the linearization cannot see, and Van der Pol from a single point. Trajectories are integrated with a fine RK4 in `f64`, with inputs that jump every sub-step. They must stay inside the time-point set at every step end and inside the time-interval set at every sub-step. For linear dynamics, a step must give the same sets as `LinearReach`. Removing the quadratic error or the Hessian spread makes a test fail. Two more tests check that an oversized set is refused, and that splitting carries a set that one piece cannot.

## Roadmap

1. Keep the second-order error as a polynomial in the set: the static part of Althoff (2013), evaluated on `R(t_k)` and added with its dependencies. Today the error enters the linear step as a box.
2. A third-order Lagrange remainder for dynamics that supply third derivatives. Today the remainder comes from the spread of the Hessian over the set.
3. A cross-check against CORA's `linearSys` reach (it runs under Octave), and a generator script for the nonlinear case. That one needs MATLAB, because CORA's symbolic derivatives do not run under Octave.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/sunsided/polyzonotope/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/sunsided/polyzonotope/blob/main/LICENSE-MIT) at your option.

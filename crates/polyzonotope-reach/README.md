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

`NonlinearReach` propagates `x' = f(x, u)` for any `Dynamics`: a type that encloses `f`, its Jacobian and its Hessians over boxes in interval arithmetic, and optionally its third derivatives. `Polynomial` implements it for polynomial vector fields and derives all of them itself. `NonlinearSettings::method` picks how each step abstracts `f`.

### Conservative linearization

`Method::Linearization` (Althoff, Stursberg & Buss 2008) is the default, since it needs the Hessians only:

1. Linearize at `z* = (x*, c_u)`, where `x*` is the middle of the set moved half a step along `f`. This gives `x' = A x + B u + w`.
2. Run the linear step with the abstraction error as an extra input, bounded by a guessed box `Ψ̄`.
3. Bound the true error over the time-interval set `R(τ)` and `U`. By Taylor's theorem with the Lagrange remainder, it is `Δzᵀ Q Δz` plus a remainder. `Q` is the center of `½ H` over a box that holds every segment from `z*` into `R(τ) × U`. The quadratic part is a polynomial zonotope (`quadratic_map`), so its hull sees `x₀²` as non-negative. The spread of the Hessian and of the Jacobian go through intervals.
4. Accept the step if the error lies in `Ψ̄`. Otherwise widen `Ψ̄` and retry. After `max_iterations` tries, refuse with `ReachError::AbstractionDiverged`.

### Conservative polynomialization

`Method::Polynomialization` (Althoff 2013) keeps the second-order error at the start set as a polynomial zonotope over the set's factors. It needs `Dynamics::third_derivatives`, and refuses with `ReachError::MissingDerivatives` without them. With `s = (x_k − x_m, 0)` the start of a trajectory relative to the middle `x_m` of the set, and `δ(t) = Δz(t) − s` what changes during the step, the error splits exactly into:

- the static part `sᵀ Q s` with `Q = ½ H(z*)`. It depends on `x_k` alone, so it is computed once per step with `quadratic_map` over the factors of `X`. It enters as the constant input `v` of the augmented system `x' = A x + v, v' = 0`, started from `X × Ψ_s`, and keeps its dependencies.
- the dynamic part: `sᵀ (½H) δ + δᵀ (½H) s + δᵀ (½H) δ`, a third-order Lagrange remainder `⅙ Σ ∂³f(ξ) Δz Δz Δz` over a box that holds every segment from `z*`, and the rounding spreads of `J(z*)` and `H(z*)`. All of it goes through interval arithmetic and into the guessed box `Ψ̄`. The state part of `δ` comes from the flow, `[0, r] f(□R(τ) × □U)` minus the half step.

The inputs are never static, since `u(t)` may jump inside `U` during the step: every term with `u − u*` is dynamic. The static error doubles the degree of the set with each step, so monomials above `max_degree` (default 4) are enclosed as independent generators. The rustdoc of `Method::Polynomialization` has the full soundness argument.

### Keeping dependencies through reduction

The linear step and the order reduction make generators independent: the error box, the time-interval hull, and every dependent monomial that reduction encloses. In the next step's quadratic map, products of independent generators are enclosed one by one, so a wide set loses its shape step by step. With `Restructure::Lift` (the default), every independent generator of the time-point set becomes a dependent factor of its own after the Girard reduction. That is exact, and later quadratic maps then treat those factors as polynomials. It costs 2 to 3 times as much per step. `Restructure::Off` keeps the old behavior.

### Adaptive steps

`NonlinearSettings::adaptive` takes an `Adaptive` with `min_step` and `max_step`. When a time step is refused, the whole time step is retried with a step `shrink` (default ½) times as long, and pieces are split only at `min_step`. After a time step whose pieces confirmed their error within `easy_guesses` (default 3) guesses, the next step is `grow` (default 1.25) times longer. Every `Step` records its `duration`, and `reach_for` runs until the durations add up to a horizon. Each step is sound for whatever length it uses, and refused attempts leave no set behind; the rustdoc of `Adaptive` has the argument.

Shorter steps are not always better. Each step adds a reduction, and on the damped cubic of the tests, steps of 0.05 diverge where steps of 0.1 do not. Growing steps also made Van der Pol's sets wider. So set `max_step` to the step that works, and let `Adaptive` shrink only where that step fails.

### Van der Pol from wide boxes

From `[1.4, 2.4] ± ρ` over 0.8 s, with 12 Taylor terms, order 20 and up to 8 pieces, the width is the sum of the interval-hull widths of the widest time-point piece. "True" is the hull of a 41 × 41 grid of RK4 trajectories. "Proven" is the time up to which the sets are proven.

| `ρ` | setting | linearization: proven, width at 0.16 s | polynomialization: proven, width at 0.16 s / 0.4 s / 0.8 s |
|-----|---------|-----------------------------|-----------------------------------------|
| 0.5 | `Off`, step 0.02 | 0.48 s, 3.71 | 0.8 s, 3.41 / 4.16 / 15.6 |
| 0.5 | `Lift`, step 0.02 | 0.8 s, 3.51 | 0.8 s, 3.39 / 3.87 / 4.20 |
| 0.5 | `Lift`, step 0.01 | 0.8 s, 3.44 | 0.8 s, 3.24 / 3.36 / 2.71 |
| 0.5 | true | | 2.74 / 2.30 / 1.09 |
| 1 | `Off`, step 0.02 | 0.22 s, 34.4 | 0.34 s, 9.91 / — / — |
| 1 | `Lift`, step 0.02 | 0.26 s, 13.9 | 0.40 s, 9.39 / 123 / — |
| 1 | `Lift`, step 0.02, adaptive down to 0.001 | 0.29 s, 13.9 | 0.44 s, 9.39 / — / — |
| 1 | `Lift`, step 0.01 | 0.33 s, 11.3 | 0.50 s, 8.48 / 26.0 / — |
| 1 | `Lift`, step 0.01, adaptive down to 0.001 | 0.34 s, 11.3 | 0.52 s, 8.48 / — / — |
| 1 | true | | 5.34 / 4.66 / 2.47 |

From `ρ = 1`, neither method gets far: the error box of a set that wide feeds on itself. Lifting and shorter steps push the failure out from 0.34 s to 0.52 s with polynomialization, and from 0.22 s to 0.34 s with linearization, and cut the width at 0.16 s from 34.4 (linearization, before) to 8.48 against a true 5.34. From `ρ = 0.5` with lifting and step 0.01, polynomialization's set shrinks again after 0.4 s, as the true set does, though at 2.5 times its width. Lifting also makes linearization's quadratic error exact on the lifted factors, so the two methods end within 1% of each other from `ρ = 0.2`.

### Splitting and merging

Both sets are reduced to the configured order after every step. `reach` runs a number of steps and `reach_for` a duration. A piece that diverges is split in two and retried, up to `max_pieces` pieces per step. The split cuts the factor of the piece, or the axis of its oriented box, whose halves have the smallest second-order error bound `Σ_i |r|ᵀ |H_i| |r|`. Factor splits keep the dependencies. The independent generators are boxed and turned into factors first, so that a factor split can reach the bulk of the set. Box splits halve the whole extent along an axis. On Van der Pol and the cubic drift, each kind wins about half of the splits. `Splitting::Factors` allows factor splits only, which keeps every dependency but needs more pieces: 30 instead of 4 for the cubic drift from radius 0.5.

After each step, two pieces merge into one when the hull of their merged enclosure is at most `merge_tolerance` (default 10%) larger by volume than their two hulls together. The merged set is also never wider than half the narrowest piece that has diverged so far. The merge encloses `λ a + (1 − λ) b` as `½ (a + b) + ν ½ (a − b)` with a fresh factor `ν`. The factors of the two pieces are paired, so two halves of a factor split merge back into their parent exactly.

```rust
use nalgebra::{DMatrix, DVector};
use polyzonotope::{SparsePolyZonotope, Zonotope};
use polyzonotope_reach::{Method, NonlinearReach, NonlinearSettings, Polynomial};

// Van der Pol: x₀' = x₁, x₁' = (1 − x₀²) x₁ − x₀.
let vdp = Polynomial::autonomous(2)
    .term(0, 1.0, &[0, 1])
    .term(1, 1.0, &[0, 1])
    .term(1, -1.0, &[2, 1])
    .term(1, -1.0, &[1, 0]);
let settings = NonlinearSettings {
    method: Method::Polynomialization,
    ..NonlinearSettings::new(0.02)
};
let reach = NonlinearReach::new(vdp, settings);
let start = SparsePolyZonotope::from_zonotope(&Zonotope::new(
    DVector::from_vec(vec![1.4, 2.4]),
    DMatrix::from_diagonal_element(2, 2, 0.05),
));

let flowpipe = reach.reach_autonomous(&start, 50)?;
assert_eq!(flowpipe.len(), 50);
# Ok::<(), polyzonotope_reach::ReachFailure>(())
```

A refusal means that no set was proven for that step. The crate does not hand back a set it could not confirm. `reach` returns a `ReachFailure` then, which holds the steps proven before the failing one and the `ReachError` that stopped it. `ReachError` is `#[non_exhaustive]`.

## Soundness

The over-approximation is sound in real arithmetic. The Taylor remainder, the time-interval corrections and the input response are all enclosed, so the sets are larger than the true reachable set, never smaller.

For nonlinear systems, the argument is the usual one for the error check. If the error over `R(τ)` lies in `Ψ̄`, no trajectory can leave `R(τ)` during the step. So the linear model plus an input in `Ψ̄` covers every trajectory. Polynomialization bounds `x(t) − x_k` by the flow over `R(τ)`, which rests on the same hypothesis. Splits are exact, and a merge encloses the convex hull of both pieces.

With the `sound` feature, the rounding of the floating-point computation is covered as well. The series terms, the factorials and the remainder are computed in outward-rounded interval arithmetic, and every set operation tracks its own rounding errors. The nonlinear step evaluates `f` and its derivatives in interval arithmetic. The model offset and the error terms are also computed that way, so they are rounded outward too. The one exception is the coefficient `i^{-i/(i-1)} − i^{-1/(i-1)}` of the time-interval correction. It comes from `libm::pow`, which is not correctly rounded, so it is widened by a relative `1e-12` instead.

## Verification

The linear tests compare against reference trajectories: matrix exponentials of the augmented system, and inputs that jump inside `U`. They check the state at the end of each step and at sampled times inside it, against both the interval hull and the support function in random directions. Without inputs, the time-point set must equal `e^{Ar} X` at matching factor values. Removing the Taylor remainder, `F`, `F̃` or `R_p` makes at least one test fail.

The nonlinear tests run every scenario with both methods: Van der Pol with an input, a pendulum with a torque input, a cubic drift that the linearization cannot see, Van der Pol from a single point, a spring whose stiffness is the input (`∂²f/∂x∂u ≠ 0`), and a damped cubic whose pieces split and later merge. Trajectories are integrated with a fine RK4 in `f64`. Every third holds the input at its upper end, every third at its lower end, and the rest jump every sub-step. They must stay inside the time-point set at every step end and inside the time-interval set at every sub-step. For linear dynamics, a step must give the same sets as `LinearReach`. Further tests check that an oversized set is refused, that a degree-9 field that overflows `f64` is refused instead of panicking, that missing third derivatives are refused at once, that the degree stays capped over 60 steps, that splitting carries a set that one piece cannot, that a failed run keeps its proven steps, that lifting keeps a wide set narrow, and that adaptive steps shrink where a fixed step fails and grow after easy steps. The Monte Carlo checks integrate each step over its own duration.

In-crate mutation tests drop one error term at a time, each in a scenario built for it, and show a trajectory escaping: the quadratic error and the Hessian spread for linearization, and the static set, the cross term, `δᵀ H δ`, the remainder and the input part of `δ` for polynomialization. The spreads of `J(z*)` and `H(z*)` come from rounding only, and no sampling can show them.

## Roadmap

1. Merge dependent monomials of small magnitude instead of boxing them (the PCA and Girard variants of Kochdumper's thesis), and an order that adapts per step. Lifting alone leaves Van der Pol from `ρ = 1` diverging after half a second.
2. A cubic map for the third-order terms at `z*`, so that the static part of the remainder also stays polynomial.
3. A cross-check against CORA's `linearSys` reach (it runs under Octave), and a generator script for the nonlinear case. That one needs MATLAB, because CORA's symbolic derivatives do not run under Octave.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/sunsided/polyzonotope/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/sunsided/polyzonotope/blob/main/LICENSE-MIT) at your option.

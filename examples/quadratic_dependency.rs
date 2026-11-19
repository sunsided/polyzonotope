//! A curved set no zonotope can represent exactly: { (α, α²) | α ∈ [-1, 1] }.
//! Run: `cargo run --example quadratic_dependency`.

use nalgebra::{DMatrix, DVector};
use polyzonotope::{FactorId, SparsePolyZonotope};

fn main() {
    let parabola = SparsePolyZonotope::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]), // columns: (1,0) for α, (0,1) for α²
        DMatrix::zeros(2, 0),
        DMatrix::from_row_slice(1, 2, &[1, 2]),
        vec![FactorId::fresh()],
    )
    .unwrap();

    println!("interval hull:      {:?}", parabola.interval_hull());
    let rotated = parabola.linear_map(&DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]));
    println!("after 90° rotation: {:?}", rotated.interval_hull());

    let (left, right) = parabola.split(parabola.ids()[0]);
    println!("left half hull:     {:?}", left.interval_hull());
    println!("right half hull:    {:?}", right.interval_hull());
}

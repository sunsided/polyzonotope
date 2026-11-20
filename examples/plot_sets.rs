//! Plots the parabola arc `{ (α, α²) }` three ways: its single zonotope
//! enclosure, the zonotope enclosures of 6 split pieces, and the arc itself.
//! Run: `cargo run --example plot_sets`, then open `target/plots/sets.svg`.

use nalgebra::{DMatrix, DVector};
use plotters::prelude::*;
use polyzonotope::{FactorId, SparsePolyZonotope, Zonotope};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parabola = SparsePolyZonotope::new(
        DVector::zeros(2),
        DMatrix::identity(2, 2), // (1, 0) for α, (0, 1) for α²
        DMatrix::zeros(2, 0),
        DMatrix::from_row_slice(1, 2, &[1, 2]),
        vec![FactorId::fresh()],
    )?;

    std::fs::create_dir_all("target/plots")?;
    let root = SVGBackend::new("target/plots/sets.svg", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .build_cartesian_2d(-1.2f64..1.2f64, -0.6f64..1.2f64)?;

    let outline = |z: &Zonotope| {
        z.polygon()
            .into_iter()
            .map(|[x, y]| (x, y))
            .collect::<Vec<_>>()
    };
    chart.draw_series([Polygon::new(
        outline(&parabola.to_zonotope()),
        BLUE.mix(0.15),
    )])?;
    chart.draw_series(
        parabola
            .split_pieces(5)
            .iter()
            .map(|piece| Polygon::new(outline(&piece.to_zonotope()), RED.mix(0.35))),
    )?;
    chart.draw_series(LineSeries::new(
        (0..=200).map(|i| {
            let p = parabola.evaluate(&[f64::from(i) / 100.0 - 1.0], &[]);
            (p[0], p[1])
        }),
        &BLACK,
    ))?;

    root.present()?;
    println!("wrote target/plots/sets.svg");
    Ok(())
}

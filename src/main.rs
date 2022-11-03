mod baseline_single;
mod common;

use std::time;

use nalgebra as na;

use common::{BETA, DIM_D, DIM_M1, DIM_M2, L};

fn main() {
    let x: na::DMatrix<f32> = na::DMatrix::new_random(DIM_M1, DIM_D);
    let y: na::DMatrix<f32> = na::DMatrix::new_random(DIM_M2, DIM_D);

    let t_start = time::Instant::now();
    let z_amm = baseline_single::beta_coocurring_amm(&x, &y, BETA, L);
    let t_amm = t_start.elapsed();

    let t_start = time::Instant::now();
    let z = x * y.transpose();
    let t_reg = t_start.elapsed();

    let e = (z - z_amm).apply_norm(&na::EuclideanNorm);

    println!(
        "B-Coocurring-AMM: Error {}; Time taken {:?}  (Regular {:?})",
        e, t_amm, t_reg
    );
}

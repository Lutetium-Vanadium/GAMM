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
    let z_loops = {
        let mut res: na::DMatrix<f32> = na::DMatrix::zeros(DIM_M1, DIM_M2);
        for i in 0..DIM_M1 {
            for j in 0..DIM_M2 {
                res[(i, j)] = x
                    .row(i)
                    .iter()
                    .zip(y.row(j).iter())
                    .map(|(&x, &y)| x * y)
                    .sum();
            }
        }
        res
    };
    let t_loops = t_start.elapsed();

    let t_start = time::Instant::now();
    let z = x * y.transpose();
    let t_lib = t_start.elapsed();

    let e_1 = (z_loops - &z).apply_norm(&na::EuclideanNorm);
    let e = (z - z_amm).apply_norm(&na::EuclideanNorm);

    println!(
        "B-Coocurring-AMM: Error {}; Time taken {:?}  (Lib {:?} [Error: {}]; Loops {:?})",
        e, t_amm, t_lib, e_1, t_loops,
    );
}

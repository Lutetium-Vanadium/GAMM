use gamm::{
    baseline_single, basic_multi,
    common::{self, BETA, L},
};
use std::time::Duration;

use nalgebra as na;

fn main() {
    let (x, y) = gamm::load_matrices().expect("Couldn't load matrices");

    // println!("x: {}", x.fixed_slice::<2, 2>(0, 0));
    // println!("y: {}", y.fixed_slice::<2, 2>(0, 0));
    let measure_loop_mm = std::env::var("MEASURE_LOOP_MM").is_ok();

    let (z_amm_single, t_amm_single) =
        gamm::measure_time(|| baseline_single::beta_coocurring_amm(&x, &y, BETA, L));
    let (z_amm_multi, t_amm_multi) =
        gamm::measure_time(|| basic_multi::beta_coocurring_amm(&x, &y, BETA, L));
    let loops_res = measure_loop_mm.then(|| {
        gamm::measure_time(|| {
            let dim_m1 = x.nrows();
            let dim_m2 = y.nrows();
            let mut res: na::DMatrix<f32> = na::DMatrix::zeros(dim_m1, dim_m2);
            for i in 0..dim_m1 {
                for j in 0..dim_m2 {
                    res[(i, j)] = x
                        .row(i)
                        .iter()
                        .zip(y.row(j).iter())
                        .map(|(&x, &y)| x * y)
                        .sum();
                }
            }
            res
        })
    });
    let (z_lib, t_lib) = gamm::measure_time(|| x * y.transpose());
    let (z, t_loops) = loops_res.unwrap_or((z_lib, Duration::MAX));

    println!("z: {}", z.fixed_slice::<2, 2>(0, 0));
    println!("z_amm_single: {}", z_amm_single.fixed_slice::<2, 2>(0, 0));
    println!("z_amm_multi: {}", z_amm_multi.fixed_slice::<2, 2>(0, 0));

    let e_amm_single = common::find_l2_norm(z.clone() - &z_amm_single);
    let e_amm_multi = common::find_l2_norm(z - &z_amm_multi);
    let e_btwn_amm = common::find_l2_norm(z_amm_single - z_amm_multi);

    if measure_loop_mm {
        println!("Loop-MM -- {:?}", t_loops);
    }

    let nthreads = common::hardware_concurrency();
    println!("Lib-MM -- {:?}", t_lib);
    println!(
        "B-Coocurring-AMM (single):\tError {}; Time taken {:?}",
        e_amm_single, t_amm_single
    );
    println!(
        "B-Coocurring-AMM (multi; {}):\tError {}; Time taken {:?}",
        nthreads, e_amm_multi, t_amm_multi
    );
    println!(
        "Multi [{}] vs Single: Error {}; Performance {:?}x",
        nthreads,
        e_btwn_amm,
        t_amm_single.as_secs_f64() / t_amm_multi.as_secs_f64()
    );
}

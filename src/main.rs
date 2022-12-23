use gamm::{baseline_single, basic_multi, common, jts_multi, self_svd_single};
use std::time::Duration;

use nalgebra as na;

fn main() {
    let config = gamm::get_config();

    let (x, y) = gamm::load_matrices(&config).expect("Couldn't load matrices");

    let measure_loop_mm = std::env::var("MEASURE_LOOP_MM").is_ok();

    let (z_amm_baseline_single, t_amm_baseline_single) = gamm::measure_time(
        || baseline_single::beta_coocurring_amm(&x, &y, &config),
        "baseline single",
    );

    let (z_amm_selfsvd_single, t_amm_selfsvd_single) = gamm::measure_time(
        || self_svd_single::beta_coocurring_amm(&x, &y, &config),
        "self svd single",
    );

    let (z_amm_libsvd_multi, t_amm_libsvd_multi) = gamm::measure_time(
        || basic_multi::beta_coocurring_amm(&x, &y, &config),
        "lib svd multi",
    );

    let (z_amm_selfsvd_multi, t_amm_selfsvd_multi) = gamm::measure_time(
        || jts_multi::beta_coocurring_amm(&x, &y, &config),
        "self svd multi",
    );

    let loops_res = measure_loop_mm.then(|| {
        gamm::measure_time(
            || {
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
            },
            "loop mm",
        )
    });
    let (z_lib, t_lib) = gamm::measure_time(|| x * y.transpose(), "lib mm");
    let (z, t_loops) = loops_res.unwrap_or((z_lib, Duration::MAX));

    println!("z: {}", z.fixed_slice::<2, 2>(0, 0));
    println!(
        "z_amm_single(lib svd): {}",
        z_amm_baseline_single.fixed_slice::<2, 2>(0, 0)
    );
    println!(
        "z_amm_single(self svd): {}",
        z_amm_selfsvd_single.fixed_slice::<2, 2>(0, 0)
    );
    println!(
        "z_amm_multi(lib svd): {}",
        z_amm_libsvd_multi.fixed_slice::<2, 2>(0, 0)
    );
    println!(
        "z_amm_multi(self svd): {}",
        z_amm_selfsvd_multi.fixed_slice::<2, 2>(0, 0)
    );

    let e_amm_baseline_single = common::find_l2_norm(&z - z_amm_baseline_single);
    let e_amm_selfsvd_single = common::find_l2_norm(&z - z_amm_selfsvd_single);
    let e_amm_libsvd_multi = common::find_l2_norm(&z - z_amm_libsvd_multi);
    let e_amm_selfsvd_multi = common::find_l2_norm(&z - z_amm_selfsvd_multi);

    if measure_loop_mm {
        println!("Loop-MM -- {:?}", t_loops);
    }

    println!("Lib-MM -- {:?}", t_lib);
    println!(
        "B-Coocurring-AMM (single lib svd):\tError {}; Time taken {:?}",
        e_amm_baseline_single, t_amm_baseline_single
    );
    println!(
        "B-Coocurring-AMM (single self svd):\tError {}; Time taken {:?}",
        e_amm_selfsvd_single, t_amm_selfsvd_single
    );
    println!(
        "B-Coocurring-AMM (multi lib svd; {}):\t\tError {}; Time taken {:?}",
        config.t, e_amm_libsvd_multi, t_amm_libsvd_multi
    );
    println!(
        "B-Coocurring-AMM (multi self svd; {}):\t\tError {}; Time taken {:?}",
        config.t, e_amm_selfsvd_multi, t_amm_selfsvd_multi
    );
}

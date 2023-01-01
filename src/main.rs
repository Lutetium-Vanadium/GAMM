use gamm::common::{self, Float};

use nalgebra as na;

fn loop_mm(
    x: &na::DMatrix<Float>,
    y: &na::DMatrix<Float>,
    _config: &gamm::config::Config,
) -> na::DMatrix<Float> {
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
}

fn main() {
    let config = gamm::get_config();
    let (x, y) = gamm::load_matrices(&config).expect("Couldn't load matrices");

    // All the ways to calculate the multiplication of x*y^T:
    let mut functions = vec![
        (
            "lib svd single",
            gamm::libsvd::single::beta_coocurring_amm as gamm::AmmFn,
        ),
        ("lib svd multi", gamm::libsvd::multi::beta_coocurring_amm),
        ("self svd single", gamm::single::beta_coocurring_amm),
        ("intra-parallelism", gamm::intra::beta_coocurring_amm),
        ("inter-parallelism", gamm::inter::beta_coocurring_amm),
        ("lib-mm", |x, y, _| x * y.transpose()),
    ];

    // If this env variable exists, then we want to measure the performance of loop based matrix
    // full multiplication.
    if std::env::var("MEASURE_LOOP_MM").is_ok() {
        functions.push(("loop-mm", loop_mm));
    }

    let (err_baseline_name, err_baseline_f) = functions.pop().unwrap();
    let (z_expected, time) = gamm::measure_time(|| err_baseline_f(&x, &y, &config));

    println!("{} -- {:?}", err_baseline_name, time);

    // Get max-width of all the names, so that the names can nicely be aligned in the output
    let max_width = functions
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0);

    for (name, f) in functions {
        let (z, time) = gamm::measure_time(|| f(&x, &y, &config));
        let err = common::find_l2_norm(&z_expected - z);

        println!(
            "{:>width$}:  Error {:.4};  Time taken {:?}",
            name,
            err,
            time,
            width = max_width
        );
    }
}

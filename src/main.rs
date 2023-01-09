#![feature(result_option_inspect)]

use std::path;

use gamm::{
    common::{self, Float},
    config,
    energy_meter::JetsonEnergyMeter,
};

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

const SINGLE: (&str, gamm::AmmFn) = ("self svd single", gamm::single::beta_coocurring_amm);
const INTRA: (&str, gamm::AmmFn) = ("intra-parallelism", gamm::intra::beta_coocurring_amm);
const INTER: (&str, gamm::AmmFn) = ("inter-parallelism", gamm::inter::beta_coocurring_amm);

fn main() {
    let config = gamm::get_config();

    // All the ways to calculate the multiplication of x*y^T:
    let mut functions = match config.bin {
        Some(config::Bin::Single) => vec![SINGLE],
        Some(config::Bin::Intra) => vec![INTRA],
        Some(config::Bin::Inter) => vec![INTER],
        None => vec![
            (
                "lib svd single",
                gamm::libsvd::single::beta_coocurring_amm as gamm::AmmFn,
            ),
            ("lib svd multi", gamm::libsvd::multi::beta_coocurring_amm),
            SINGLE,
            INTRA,
            INTER,
        ],
    };

    functions.push(("lib-mm", |x, y, _| x * y.transpose()));

    // If this env variable exists, then we want to measure the performance of loop based matrix
    // full multiplication.
    if std::env::var("MEASURE_LOOP_MM").is_ok() {
        functions.push(("loop-mm", loop_mm));
    }

    run_functions(functions, config);
}

/// Run the given `functions` and measure time taken, energy consumed and error for the given AMM.
///
/// The last function is taken to be the baseline from which the error is calculated.
fn run_functions(mut functions: Vec<(&str, gamm::AmmFn)>, config: config::Config) {
    let (x, y) = gamm::load_matrices(&config).expect("Couldn't load matrices");

    let (err_baseline_name, err_baseline_f) = functions.pop().unwrap();
    let (z_expected, time) = gamm::measure_time(|| err_baseline_f(&x, &y, &config));

    println!("(baseline) {err_baseline_name} -- {time:?}");

    // Try to use the energy meter. If there is an error in creating the energy meter, then the
    // energy meter will not be supported
    let mut energy_meter = JetsonEnergyMeter::new()
        .inspect(|_| eprintln!("Using Jetson Energy Meter"))
        .ok();

    // Get max-width of all the names, so that the names can nicely be aligned in the output
    let max_width = functions
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0);

    // If set, all the detailed energu readings generated will be stored as csvs in this directory
    let mut energy_readings_dir = get_energy_readings_dir();

    for (name, f) in functions {
        // Start sampling energy readings if there is an energy meter
        energy_meter
            .as_mut()
            .map(JetsonEnergyMeter::start_sampling)
            .transpose()
            .expect("Energy meter already sampling");

        let (z, time) = gamm::measure_time(|| f(&x, &y, &config));

        // Stop sampling if there is an energy meter
        let energy_readings = energy_meter
            .as_mut()
            .map(JetsonEnergyMeter::stop_sampling)
            .transpose()
            .expect("Energy meter failed");

        if let Some((energy_readings_path, energy_readings)) =
            energy_readings_dir.as_mut().zip(energy_readings.as_ref())
        {
            // Store readings in `ENERGY_READINGS_DIR/amm_name.csv`
            energy_readings_path.push(name);
            energy_readings_path.set_extension("csv");

            eprintln!("Writing energy readings to {energy_readings_path:?}");

            if let Err(e) = energy_readings.write_csv(&energy_readings_path) {
                eprintln!("WARNING: Failed to write detailed energy readings:\n{e}");
            }

            // Remember to remove the filename added earlier, as the energy_readings_dir is
            // supposed to be the directory
            energy_readings_path.pop();
        }

        let err = common::find_l2_norm(&z_expected - z);

        print!(
            " {:>width$}:  Time - {:>8.4}s;  ",
            name,
            time.as_secs_f64(),
            width = max_width
        );

        // If there is an energy meter, then also print the energy consumed
        if let Some(ref energy_readings) = energy_readings {
            print!("Energy - {:>8.4}J;  ", energy_readings.energy_consumed());
        }

        println!("Error - {err:.4}");
    }
}

fn get_energy_readings_dir() -> Option<path::PathBuf> {
    std::env::var_os("ENERGY_READINGS_DIR").map(From::from)
}

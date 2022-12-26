use gamm::{common::Float, config};
use nalgebra as na;

type AmmFn = fn(&na::DMatrix<Float>, &na::DMatrix<Float>, &config::Config) -> na::DMatrix<Float>;

fn main() {
    let config = gamm::get_config();
    let (x, y) = gamm::load_matrices(&config).expect("Couldn't load matrices");

    let (name, f) = match config.bin.as_deref() {
        Some("intra") => ("intra", gamm::self_svd_single::beta_coocurring_amm as AmmFn),
        Some("inter") => ("inter", gamm::jts_multi::beta_coocurring_amm as AmmFn),
        _ => panic!("Invalid amm ({:?}) set to be profiled", config.bin),
    };

    let mut energy_meter =
        gamm::energy_meter::JetsonEnergyMeter::new().expect("Couldn't create energy meter");

    energy_meter
        .start_sampling()
        .expect("Energy meter couldn't have already started sampling");
    let (mat, time) = gamm::measure_time(|| f(&x, &y, &config), name);
    energy_meter.stop_sampling().expect("Energy meter failed");

    std::hint::black_box(mat);

    // Uncomment this line to store more detailed energy readings.
    // energy_meter.write_csv("energy_consumed.csv").expect("Energy meter failed");

    println!(
        "Time: {:?}; Energy: {} J",
        time,
        energy_meter
            .energy_consumed()
            .expect("Energy meter sampling has been stopped")
    )
}

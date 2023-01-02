use gamm::config::Bin;
fn main() {
    let config = gamm::get_config();
    let (x, y) = gamm::load_matrices(&config).expect("Couldn't load matrices");

    let (name, f) = match config.bin.expect("No amm set to be profiled") {
        Bin::Intra => ("intra", gamm::intra::beta_coocurring_amm as gamm::AmmFn),
        Bin::Inter => ("inter", gamm::inter::beta_coocurring_amm as gamm::AmmFn),
        Bin::Single => ("single", gamm::single::beta_coocurring_amm as gamm::AmmFn),
    };

    let mut energy_meter =
        gamm::energy_meter::JetsonEnergyMeter::new().expect("Couldn't create energy meter");

    energy_meter
        .start_sampling()
        .expect("Energy meter couldn't have already started sampling");
    let (z_amm, time) = gamm::measure_time(|| f(&x, &y, &config));
    let energy_readings = energy_meter.stop_sampling().expect("Energy meter failed");

    let z_actual = x * y.transpose();
    let err = gamm::common::find_l2_norm(z_actual - z_amm);

    if let Some(energy_readings_path) = std::env::var_os("WRITE_ENERGY_READINGS") {
        println!("Writing energy readings to {:?}", energy_readings_path);
        if let Err(e) = energy_readings.write_csv(&energy_readings_path) {
            println!("WARNING: Failed to write detailed energy readings:");
            println!("{}", e);
        }
    }

    println!("{}-parallelism", name);
    println!(
        "Time: {:?}; Energy: {} J; Error: {}",
        time,
        energy_readings.energy_consumed(),
        err
    )
}

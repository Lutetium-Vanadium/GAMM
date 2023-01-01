fn main() {
    let config = gamm::get_config();
    let (x, y) = gamm::load_matrices(&config).expect("Couldn't load matrices");

    let (z_amm, time) = gamm::measure_time(|| gamm::intra::beta_coocurring_amm(&x, &y, &config));

    let z_actual = x * y.transpose();
    let err = gamm::common::find_l2_norm(z_actual - z_amm);

    println!("intra-parllelism");
    println!("Time: {:?}; Error: {}", time, err);
}

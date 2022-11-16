fn main() {
    let config = gamm::get_config();
    let (x, y) = gamm::load_matrices(&config).expect("Couldn't load matrices");
    std::hint::black_box(gamm::basic_multi::beta_coocurring_amm(&x, &y, &config));
}

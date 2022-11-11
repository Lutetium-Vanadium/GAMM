use gamm::common::{BETA, L};

fn main() {
    let (x, y) = gamm::load_matrices().expect("Couldn't load matrices");
    std::hint::black_box(gamm::basic_multi::beta_coocurring_amm(&x, &y, BETA, L));
}

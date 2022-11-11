fn main() {
    let (x, y) = gamm::load_matrices().expect("Couldn't load matrices");
    std::hint::black_box(x * y.transpose());
}

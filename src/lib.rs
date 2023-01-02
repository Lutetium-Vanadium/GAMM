#![feature(slice_take, io_error_other)]

pub mod bamm;
pub mod common;
pub mod config;
pub mod energy_meter;
pub mod inter;
pub mod intra;
pub mod libsvd;
pub mod single;
pub mod svd;

use std::{
    fs,
    io::{self, Read},
    path, time,
};

use nalgebra as na;

pub type AmmFn = fn(
    &na::DMatrix<common::Float>,
    &na::DMatrix<common::Float>,
    &config::Config,
) -> na::DMatrix<common::Float>;

pub fn measure_time<T>(f: impl FnOnce() -> T) -> (T, time::Duration) {
    let start = time::Instant::now();
    let res = f();
    let duration = start.elapsed();
    (res, duration)
}

fn get_config_inner() -> Option<config::Config> {
    let mut args = std::env::args();

    if args.len() < 2 {
        return None;
    }

    config::Config::from_file(path::Path::new(
        &args.next_back().expect("Checked len above"),
    ))
    .ok()
    .flatten()
}

pub fn get_config() -> config::Config {
    let config = get_config_inner().unwrap_or_default();
    println!("{:?}", config);
    config
}

pub fn load_matrix(path: &path::Path) -> io::Result<na::DMatrix<common::Float>> {
    let mut r = io::BufReader::new(fs::File::open(path)?);
    let mut buf = [0u8; 8];
    let n = {
        r.read_exact(&mut buf)?;
        u64::from_le_bytes(buf) as usize
    };
    let m = {
        r.read_exact(&mut buf)?;
        u64::from_le_bytes(buf) as usize
    };

    let mut mat = na::DMatrix::zeros(n, m);

    for i in 0..n {
        for j in 0..m {
            let mut buf = [0u8; std::mem::size_of::<common::Float>()];
            r.read_exact(&mut buf)?;
            mat[(i, j)] = common::Float::from_ne_bytes(buf);
        }
    }

    Ok(mat)
}

pub fn load_matrices(
    config: &config::Config,
) -> io::Result<(na::DMatrix<common::Float>, na::DMatrix<common::Float>)> {
    Ok((load_matrix(&config.x)?, load_matrix(&config.y)?))
}

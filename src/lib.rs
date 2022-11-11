pub mod baseline_single;
pub mod basic_multi;
pub mod common;

use std::{
    fs,
    io::{self, Read},
    path, time,
};

use nalgebra as na;

pub fn measure_time<T>(f: impl FnOnce() -> T) -> (T, time::Duration) {
    let start = time::Instant::now();
    let res = f();
    (res, start.elapsed())
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

pub fn load_matrices() -> io::Result<(na::DMatrix<common::Float>, na::DMatrix<common::Float>)> {
    Ok((
        load_matrix(path::Path::new("./baseline/x.dat"))?,
        load_matrix(path::Path::new("./baseline/y.dat"))?,
    ))
}

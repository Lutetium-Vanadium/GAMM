#![allow(dead_code)]

mod baseline_single;
mod common;

use std::{
    fs,
    io::{self, Read},
    path, time,
};

use nalgebra as na;

use common::{BETA, L};

fn main() {
    let (x, y) = load_matrices().expect("Couldn't load matrices");

    let t_start = time::Instant::now();
    let z_amm = baseline_single::beta_coocurring_amm(&x, &y, BETA, L);
    let t_amm = t_start.elapsed();

    let dim_m1 = x.nrows();
    let dim_m2 = y.nrows();

    println!("x: {}", x.fixed_slice::<2, 2>(0, 0));
    println!("y: {}", y.fixed_slice::<2, 2>(0, 0));

    let t_start = time::Instant::now();
    let z_loops = {
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
    };
    let t_loops = t_start.elapsed();

    let t_start = time::Instant::now();
    let z_lib = x * y.transpose();
    let t_lib = t_start.elapsed();

    println!("z_amm: {}", z_amm.fixed_slice::<2, 2>(0, 0));
    println!("z_lib: {}", z_lib.fixed_slice::<2, 2>(0, 0));
    println!("z_loops: {}", z_loops.fixed_slice::<2, 2>(0, 0));

    let e_lib = common::find_l2_norm(&z_loops - z_lib);
    let e_amm = common::find_l2_norm(z_loops - z_amm);

    println!(
        "B-Coocurring-AMM: Error {}; Time taken {:?}  (Lib {:?} [Error: {}]; Loops {:?})",
        e_amm, t_amm, t_lib, e_lib, t_loops,
    );
}

fn load_matrix(path: &path::Path) -> io::Result<na::DMatrix<common::Float>> {
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

fn load_matrices() -> io::Result<(na::DMatrix<common::Float>, na::DMatrix<common::Float>)> {
    Ok((
        load_matrix(path::Path::new("./baseline/x.dat"))?,
        load_matrix(path::Path::new("./baseline/y.dat"))?,
    ))
}

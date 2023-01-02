use nalgebra as na;

use crate::{
    common::{self, Float, ZeroedColumns},
    config, svd,
};

pub struct BAmmConfig {
    pub l: usize,
    pub attenuate_vec: na::DVector<Float>,
}

impl From<&config::Config> for BAmmConfig {
    fn from(config: &config::Config) -> Self {
        let config::Config { l, beta, .. } = *config;
        Self {
            l,
            attenuate_vec: na::DVector::from_iterator(
                config.l,
                (0..l)
                    .map(|i| (beta * (i as Float) / ((l as Float) - 1.0)).exp_m1() / beta.exp_m1()),
            ),
        }
    }
}

pub fn parameterized_reduce_rank(sv: &mut na::DVector<Float>, attenuate_vec: &na::DVector<Float>) {
    let delta = sv[0];

    sv.axpy(-delta, &attenuate_vec, 1.0);
    sv.apply(|v| *v = v.max(0.0).sqrt());
}

// &mut is used for bx and by instead of DMatrixSliceMut as it is easier to deal with calling
// functions in a loop
pub fn beta_coocurring_reduction<S: svd::SVDCalculator>(
    x: na::DMatrixSlice<Float>,
    y: na::DMatrixSlice<Float>,
    bx: &mut na::DMatrix<Float>,
    by: &mut na::DMatrix<Float>,
    svd_executor: &S,
    config: &BAmmConfig,
) {
    let (_, d1) = x.shape();
    let (_, d2) = y.shape();
    debug_assert_eq!(d1, d2);
    debug_assert!(config.l <= d1);

    let mut zeroed_cols = ZeroedColumns::new_from_matrix(&bx);
    assert_eq!(x.ncols(), y.ncols());

    // Get an iterator over the non-zero columns of X and Y
    let mut x_iter = x
        .column_iter()
        .enumerate()
        .filter(|(_, c)| !c.iter().copied().all(common::is_zero))
        .peekable(); // Needs to be peekable so that we can check if it is empty
    let mut y_iter = y
        .column_iter()
        .enumerate()
        .filter(|(_, c)| !c.iter().copied().all(common::is_zero));

    while x_iter.peek().is_some() {
        beta_coocurring_reduction_step(
            &mut x_iter,
            &mut y_iter,
            bx,
            by,
            &mut zeroed_cols,
            svd_executor,
            config,
        )
    }
}

pub fn beta_coocurring_reduction_step<'a, S: svd::SVDCalculator>(
    x_iter: &mut impl Iterator<Item = (usize, na::DVectorSlice<'a, Float>)>,
    y_iter: &mut impl Iterator<Item = (usize, na::DVectorSlice<'a, Float>)>,
    mut bx: &mut na::DMatrix<Float>,
    mut by: &mut na::DMatrix<Float>,
    zeroed_cols: &mut ZeroedColumns,
    svd_executor: &S,
    config: &BAmmConfig,
) {
    // Just make sure that the zero column book-keeping is indeed valid.
    // This does not run in release mode
    if cfg!(debug_assertions) {
        zeroed_cols.check_matching_zeroed(&bx);
        zeroed_cols.check_matching_zeroed(&by);
    }

    // No space, perform rank reduction
    if zeroed_cols.nzeroed() == 0 {
        let BAmmConfig {
            l,
            ref attenuate_vec,
        } = *config;

        let (qx, mut rx) = common::qr(bx.clone_owned(), na::DMatrix::zeros(l, l), false);
        let (qy, ry_t) = common::qr(by.clone_owned(), na::DMatrix::zeros(l, l), true);

        rx *= ry_t;

        let svd::Svd {
            mut u,
            mut v,
            mut sv,
        } = svd_executor.svd(rx);

        parameterized_reduce_rank(&mut sv, attenuate_vec);

        for i in 0..sv.len() {
            if common::is_zero(sv[i]) {
                zeroed_cols.set_zeroed(i);
            }
            u.column_mut(i).scale_mut(sv[i]);
            v.column_mut(i).scale_mut(sv[i]);
        }

        qx.mul_to(&u, &mut bx);
        qy.mul_to(&v, &mut by);
    }

    // Fill Bx and By with columns from X and Y
    for ((x_i, x_col), (y_i, y_col)) in x_iter.zip(y_iter).take(zeroed_cols.nzeroed()) {
        assert_eq!(x_i, y_i);

        let zero_col = zeroed_cols.get_next_zeroed();
        bx.set_column(zero_col, &x_col);
        by.set_column(zero_col, &y_col);
    }
}

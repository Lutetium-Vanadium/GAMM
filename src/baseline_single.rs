use nalgebra as na;
use num_traits::Zero;

use crate::{
    common::{self, Float, ZeroedColumns},
    config,
};

pub fn beta_coocurring_amm(
    x: &na::DMatrix<Float>,
    y: &na::DMatrix<Float>,
    config: &config::Config,
) -> na::DMatrix<Float> {
    let config::Config { l, beta, .. } = *config;

    let (_, d1) = x.shape();
    let (_, d2) = y.shape();
    debug_assert_eq!(d1, d2);
    let d = d1;
    debug_assert!(l <= d);

    // Start by inserting first l columns.
    //
    // The loop essentially inserts replaces zero columns with columns from x and y until there are
    // no zero columns left, so the first l columns can be copied beforehand
    let mut bx = na::DMatrix::from(x.columns(0, l));
    let mut by = na::DMatrix::from(y.columns(0, l));

    let mut zeroed_cols = ZeroedColumns::new_no_zeroed(l);

    let attenuate_vec =
        na::DVector::from_iterator(l, (0..l).map(|i| attenuate(beta, i as Float, l as Float)));

    let mut ry_t = na::DMatrix::zeros(l, l);

    // First l columns have already been copied, so we start from i=l instead
    for i in l..d {
        // In the paper's algorithm, the column insertion happens before this check as there will
        // always be non-zero rows to copy into since the previous iteration would create space if
        // needed. Since we copy the first l rows beforehand, the space making must be done before
        // subsequent copies
        if zeroed_cols.nzeroed() == 0 {
            let mut rx = na::DMatrix::zeros(l, l);

            let qx = {
                let res = common::qr(bx, rx, false);
                rx = res.1;
                res.0
            };
            let qy = {
                let res = common::qr(by, ry_t, true);
                ry_t = res.1;
                res.0
            };

            rx *= &ry_t;

            let na::SVD {
                u,
                v_t,
                singular_values: mut sv,
            } = rx.svd(true, true);

            let mut u = u.expect("true passed to compute_u");
            let mut v_t = v_t.expect("true passed to compute_v");

            parameterized_reduce_rank(&mut sv, &attenuate_vec);

            let mut v = {
                v_t.transpose_mut();
                v_t
            };

            let check_no_zero_cols =
                |m: &na::DMatrix<Float>| m.column_iter().all(|c| !c.iter().all(|x| x.is_zero()));
            // The following matrices are orthogonal, so they shouldn't have any zero columns
            // This fact is used to optimize zero column finding by keeping track of zero columns
            // using the svd
            debug_assert!(check_no_zero_cols(&u));
            debug_assert!(check_no_zero_cols(&v));
            debug_assert!(check_no_zero_cols(&qx));
            debug_assert!(check_no_zero_cols(&qy));

            for i in 0..sv.len() {
                if sv[i].is_zero() {
                    zeroed_cols.set_zeroed(i);
                }
                u.column_mut(i).scale_mut(sv[i]);
                v.column_mut(i).scale_mut(sv[i]);
            }

            bx = qx * u;
            by = qy * v;
        }

        // Just make sure that the zero column book-keeping is indeed valid
        if cfg!(debug_assertions) {
            zeroed_cols.check_matching_zeroed(&bx);
            zeroed_cols.check_matching_zeroed(&by);
        }

        let zero_col = zeroed_cols.get_next_zeroed();
        bx.set_column(zero_col, &x.column(i));
        by.set_column(zero_col, &y.column(i));
    }

    bx * by.transpose()
}

fn parameterized_reduce_rank(
    sv: &mut na::OVector<
        <Float as na::ComplexField>::RealField,
        na::DimMinimum<na::Dynamic, na::Dynamic>,
    >,
    attenuate_vec: &na::DVector<Float>,
) {
    let delta = sv[0];

    sv.axpy(-delta, attenuate_vec, 1.0);
    sv.apply(|v| *v = v.max(0.0).sqrt());
}

fn attenuate(beta: Float, k: Float, l: Float) -> Float {
    (beta * k / (l - 1.0)).exp_m1() / beta.exp_m1()
}

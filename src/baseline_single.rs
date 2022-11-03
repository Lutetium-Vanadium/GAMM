use nalgebra::{self as na, DMatrix};
use num_traits::Zero;

use crate::common::{Float, ZeroedColumns};

pub fn beta_coocurring_amm(
    x: &DMatrix<Float>,
    y: &DMatrix<Float>,
    beta: Float,
    l: usize,
) -> DMatrix<Float> {
    let (_, d1) = x.shape();
    let (_, d2) = y.shape();
    debug_assert_eq!(d1, d2);
    let d = d1;
    debug_assert!(l <= d);

    // Start by inserting first l columns.
    //
    // The loop essentially inserts replaces zero columns with columns from x and y until there are
    // no zero columns left, so the first l columns can be copied beforehand
    let mut bx: DMatrix<Float> = DMatrix::from(x.columns(0, l));
    let mut by: DMatrix<Float> = DMatrix::from(y.columns(0, l));

    let mut zeroed_cols = ZeroedColumns::new_no_zeroed(l);

    // First l columns have already been copied, so we start from i=l instead
    for i in l..d {
        // In the paper's algorithm, the column insertion happens before this check as there will
        // always be non-zero rows to copy into since the previous iteration would create space if
        // needed. Since we copy the first l rows beforehand, the space making must be done before
        // subsequent copies
        if zeroed_cols.nzeroed() == 0 {
            let (mut qx, mut rx) = bx.qr().unpack();
            let (mut qy, mut ry) = by.qr().unpack();

            // TODO check if in-place transpose is more efficient or maintaining few matrix buffers
            // in which to transpose is more efficient
            ry.transpose_mut();
            rx *= ry;

            let na::SVD {
                u,
                v_t,
                singular_values: mut sv,
            } = rx.svd(true, true);
            let mut u = u.expect("true passed to compute_u");
            let mut v_t = v_t.expect("true passed to compute_v");

            parameterized_reduce_rank(&mut sv, l, beta);

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

            for i in 0..sv.shape().0 {
                if sv[i].is_zero() {
                    zeroed_cols.set_zeroed(i);
                }
                u.column_mut(i).scale_mut(sv[i]);
                v.column_mut(i).scale_mut(sv[i]);
            }

            qx *= u;
            qy *= v;

            bx = qx;
            by = qy;
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
    l: usize,
    beta: Float,
) {
    let delta = sv[l / 2];

    // TODO: vectorize this?
    for i in 0..l {
        sv[i] = (sv[i] - delta * attenuate(beta, i as Float, l as Float))
            .max(0.0)
            .sqrt();
    }
}

fn attenuate(beta: Float, k: Float, l: Float) -> Float {
    (beta * k / (l - 1.0)).exp_m1() / beta.exp_m1()
}
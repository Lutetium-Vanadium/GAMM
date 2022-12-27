use std::sync::{Barrier, Mutex};

use nalgebra as na;

use crate::{
    common::{self, Float, ZeroedColumns},
    config, svd,
};

pub fn beta_coocurring_amm(
    x: &na::DMatrix<Float>,
    y: &na::DMatrix<Float>,
    config: &config::Config,
) -> na::DMatrix<Float> {
    let config::Config { l, beta, t, .. } = *config;

    let (m1, d1) = x.shape();
    let (m2, d2) = y.shape();
    assert_eq!(d1, d2);
    let d = d1;

    let sub_col_size_base = d / t;
    let extra = d % t;

    let attenuate_vec =
        na::DVector::from_iterator(l, (0..l).map(|i| attenuate(beta, i as Float, l as Float)));

    let barrier = Barrier::new(t);
    let matrices: Vec<_> = (0..t)
        .map(|_| Mutex::new((na::DMatrix::zeros(m1, l), na::DMatrix::zeros(m2, l))))
        .collect();

    std::thread::scope(|s| {
        let handles: Vec<_> = (0..t)
            .map(|i| {
                let barrier_ref = &barrier;
                let matrices_ref = matrices.as_ref();
                let attenuate_vec = attenuate_vec.column(0);

                let (start_i, ncols) = if i < extra {
                    (i * (sub_col_size_base + 1), sub_col_size_base + 1)
                } else {
                    (i * sub_col_size_base + extra, sub_col_size_base)
                };

                let x_slice = x.columns(start_i, ncols);
                let y_slice = y.columns(start_i, ncols);

                s.spawn(move || {
                    thread_task(
                        i,
                        t,
                        x_slice,
                        y_slice,
                        barrier_ref,
                        matrices_ref,
                        attenuate_vec,
                        l,
                    )
                })
            })
            .collect();

        let _ = handles.into_iter().map(|h| h.join());
    });

    let (bx, by) = matrices
        .into_iter()
        .next()
        .unwrap()
        .into_inner()
        .expect("Poisoned lock");

    bx * by.transpose()
}

/// # Execution
///
/// For example, take t = 4, the theoretical tree will look something like this:
/// ```text
/// 0 --.
///     +--.
/// 1 --'  |
///        +-- complete
/// 2 --.  |
///     +--'
/// 3 --'
/// ```
///
/// This is how it is performed:
/// ```text
/// 0 --+--+-- complete
/// 1 --'  |
/// 2 --+--'
/// 3 --'
/// ```
fn thread_task(
    thread_id: usize,
    t: usize,
    x: na::DMatrixSlice<Float>,
    y: na::DMatrixSlice<Float>,
    barrier: &Barrier,
    matrices: &[Mutex<(na::DMatrix<Float>, na::DMatrix<Float>)>],
    attenuate_vec: na::DVectorSlice<Float>,
    l: usize,
) {
    let d = x.ncols();

    // The maximum number of times this thread needs to run. Each parallel iteration of
    // beta_coocurring_reduction requires half the threads* of the previous iteration.
    //
    // For the case where t is not a power of 2, this division is imperfect, so for larger
    // thread_id it will stop when the index of the thread to merge with goes out of bounds
    //
    // * when t is not a power of two, this may not be true. This is handled by the bounds
    //   checking when getting the `other_matrices` in the loop.
    let nruns = (thread_id | t.next_power_of_two()).trailing_zeros() + 1;

    let mut own_matrices = matrices[thread_id]
        .try_lock()
        .expect("couldn't lock own threads matrix");

    let (ref mut bx, ref mut by) = *own_matrices;

    // Make sure every thread gets first access to its assigned matrices. If the thread execution
    // was very imbalanced, it could be that a thread completes its first run and then takes the
    // `Mutex` if this thread which would be erroneous.
    barrier.wait();

    // Start by inserting first l columns.
    //
    // The loop essentially inserts replaces zero columns with columns from x and y until there are
    // no zero columns left, so the first l columns can be copied beforehand
    bx.copy_from(&x.columns(0, l));
    by.copy_from(&y.columns(0, l));

    for i in 0..nruns {
        let other_matrices = if i > 0 {
            match matrices.get(thread_id + 2usize.pow(i - 1)) {
                Some(l) => Some(l.lock().expect("couldn't lock other thread's matrix")),
                // If t is not a power of 2, then higher thread_ids may need to merge and reduce from
                // threads that don't exist. In that case, the thread can return early as the thread to
                // merge and reduce with is strictly increasing.
                None => break,
            }
        } else {
            None
        };

        let (x, y) = other_matrices
            .as_ref()
            .map(|m| (m.0.columns(0, l), m.1.columns(0, l)))
            .unwrap_or_else(|| (x.columns(l, d - l), y.columns(l, d - l)));

        beta_coocurring_reduction(
            x,
            y,
            attenuate_vec,
            l,
            bx.columns_mut(0, l),
            by.columns_mut(0, l),
        );
    }

    drop(own_matrices);
}

/// First `l` columns must already be inserted into `bx` and `by`. `x` and `y` should not include
/// the first `l` columns
///
/// Shape of matrices:
/// - x: (m1, d-l)
/// - y: (m2, d-l)
/// - bx: (m1, l)
/// - by: (m2, l)
fn beta_coocurring_reduction(
    x: na::DMatrixSlice<Float>,
    y: na::DMatrixSlice<Float>,
    attenuate_vec: na::DVectorSlice<Float>,
    l: usize,
    mut bx: na::DMatrixSliceMut<Float>,
    mut by: na::DMatrixSliceMut<Float>,
) {
    // Need to make sure there aren't already zero columns
    let mut zeroed_cols = ZeroedColumns::new_from_matrix(&bx);
    assert_eq!(x.ncols(), y.ncols());

    let mut ry_t = na::DMatrix::zeros(l, l);

    let x_iter = x
        .column_iter()
        .enumerate()
        .filter(|(_, c)| !c.iter().copied().all(common::is_zero));
    let y_iter = y
        .column_iter()
        .enumerate()
        .filter(|(_, c)| !c.iter().copied().all(common::is_zero));

    // First l columns have already been copied, so we start from i=l instead
    for ((x_i, x_col), (y_i, y_col)) in x_iter.zip(y_iter) {
        assert_eq!(x_i, y_i);
        // In the paper's algorithm, the column insertion happens before this check as there will
        // always be non-zero rows to copy into since the previous iteration would create space if
        // needed. Since we copy the first l rows beforehand, the space making must be done before
        // subsequent copies
        if zeroed_cols.nzeroed() == 0 {
            let mut rx = na::DMatrix::zeros(l, l);

            let qx = {
                // TODO: try avoiding this allocation + copy?
                let res = common::qr(bx.clone_owned(), rx, false);
                rx = res.1;
                res.0
            };
            let qy = {
                // TODO: try avoiding this allocation + copy?
                let res = common::qr(by.clone_owned(), ry_t, true);
                ry_t = res.1;
                res.0
            };

            // TODO: try avoiding this allocation?
            rx *= &ry_t;

            let svd::Svd {
                mut u,
                mut v,
                mut sv,
            } = svd::Svd::jts_seq(rx, 1e-7, 32, 100);

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

        // Just make sure that the zero column book-keeping is indeed valid - only run in debug mode
        if cfg!(debug_assertions) {
            zeroed_cols.check_matching_zeroed(&bx);
            zeroed_cols.check_matching_zeroed(&by);
        }

        let zero_col = zeroed_cols.get_next_zeroed();
        bx.set_column(zero_col, &x_col);
        by.set_column(zero_col, &y_col);
    }
}

fn parameterized_reduce_rank(
    sv: &mut na::OVector<
        <Float as na::ComplexField>::RealField,
        na::DimMinimum<na::Dynamic, na::Dynamic>,
    >,
    attenuate_vec: na::DVectorSlice<Float>,
) {
    let delta = sv[0];

    sv.axpy(-delta, &attenuate_vec, 1.0);
    // TODO vectorize this?
    sv.apply(|v| *v = v.max(0.0).sqrt());
}

fn attenuate(beta: Float, k: Float, l: Float) -> Float {
    (beta * k / (l - 1.0)).exp_m1() / beta.exp_m1()
}

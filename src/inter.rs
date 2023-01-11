use std::sync::{Barrier, Mutex};

use nalgebra as na;

use scoped_pool::Pool;

use crate::{
    bamm,
    common::{self, Float},
    config, svd,
};

pub fn beta_coocurring_amm(
    x: &na::DMatrix<Float>,
    y: &na::DMatrix<Float>,
    config: &config::Config,
) -> na::DMatrix<Float> {
    let config::Config { l, t, .. } = *config;

    let (m1, d1) = x.shape();
    let (m2, d2) = y.shape();
    assert_eq!(d1, d2);
    let d = d1;

    if t * l > d {
        return na::DMatrix::zeros(m1, m2);
    }

    let bamm_config = bamm::BAmmConfig::from(config);
    let barrier = Barrier::new(t);
    let matrices: Vec<_> = (0..t)
        .map(|_| Mutex::new((na::DMatrix::zeros(m1, l), na::DMatrix::zeros(m2, l))))
        .collect();

    // create a pool of t threads
    let pool = Pool::new(t);

    pool.scoped(|s| {
        for i in 0..t {
            let barrier_ref = &barrier;
            let matrices_ref = matrices.as_ref();
            let bamm_config_ref = &bamm_config;

            let (start_col_i, ncols) = common::uneven_divide(i, d, t);

            let x_slice = x.columns(start_col_i, ncols);
            let y_slice = y.columns(start_col_i, ncols);

            s.execute(move || {
                thread_task(
                    i,
                    t,
                    x_slice,
                    y_slice,
                    barrier_ref,
                    matrices_ref,
                    bamm_config_ref,
                )
            })
        }
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
    bamm_config: &bamm::BAmmConfig,
) {
    let l = bamm_config.l;

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
            .unwrap_or_else(|| (x.columns_range(l..), y.columns_range(l..)));

        bamm::beta_coocurring_reduction(x, y, bx, by, &svd::SeqJTSConfig::default(), bamm_config);
    }

    drop(own_matrices);
}

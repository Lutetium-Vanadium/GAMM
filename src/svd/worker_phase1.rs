use nalgebra as na;

use super::WorkerArgs;
use crate::common::Float;

#[cfg_attr(feature = "profile", inline(never))]
#[cfg_attr(not(feature = "profile"), inline(always))]
pub(super) unsafe fn jts_group_worker_phase_1<D: na::Dim>(
    worker_id: usize,
    t: usize,
    args: &WorkerArgs<'_, D>,
) where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    let n = args.n;

    // ------------ PHASE 1.1 ------------
    // Generate all the possible column pairs (in parallel).
    {
        // SAFETY: all worker threads are the in phase 1.1. Here b is read-only, so casting the b
        // pointer to a MatrixSlice is ok.
        //
        // The pointer is also valid for n*n matrix (guaranteed by the caller)
        let b_slice = unsafe { std::slice::from_raw_parts(args.b.as_ptr(), n * n) };
        let b = na::MatrixSlice::from_slice_generic(b_slice, args.shape.0, args.shape.1);
        for j in (worker_id..(n - 1)).step_by(t) {
            for k in (j + 1)..n {
                // SAFETY: caller guarantees p points to a pre-allocated buffer which is writeable.
                // Each index here is unique and so writing to it in parallel is ok.
                unsafe {
                    args.p
                        .as_ptr()
                        // Each iteration (j) has (n-(j+1)) inner iterations. Current index will be
                        // sum of number of previous inner iterations plus number of inner
                        // iterations done for this time.
                        //
                        //   vvvvvvvvvvvvvvvvvvvvvvvvvvv-- number of previous inner iterations
                        .add((j * (2 * n - (j + 1)) / 2) + (k - (j + 1)))
                        //             number of current --^^^^^^^^^^^^^
                        //              inner iterations
                        .write((j, k, b.column(j).dot(&b.column(k))))
                };
            }
        }
    }

    args.barrier.wait(); // Wait for p to be filled by all threads. Only then commence sorting

    // ------------ PHASE 1.2 ------------
    // Sort the column-pairs by dot-product. This is performed only by the main worker.

    if worker_id == super::MAIN_WORKER {
        // SAFETY: caller guarantees p points to a pre-allocated buffer which is npivots long. The
        // values of p were written already in phase 1.1
        let p = unsafe { std::slice::from_raw_parts_mut(args.p.as_ptr(), args.npivots) };

        // Sort by descending order of dot-products
        p.sort_unstable_by(|(_, _, a), (_, _, b)| {
            b.partial_cmp(&a).expect("Singular value was NaN")
        });

        // Barrier takes care of synchronisation, so relaxed ordering is fine
        args.counter.store(
            if p[0].2 < args.delta {
                super::COMPLETED
            } else {
                super::GENERATING
            },
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

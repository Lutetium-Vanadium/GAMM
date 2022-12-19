use std::sync::atomic;

use nalgebra as na;

use super::{JacobiRotation, WorkerArgs};
use crate::common::Float;

#[cfg_attr(feature = "profile", inline(never))]
#[cfg_attr(not(feature = "profile"), inline(always))]
pub(super) unsafe fn jts_group_worker_phase_2<D: na::Dim>(
    worker_id: usize,
    t: usize,
    args: &WorkerArgs<'_, D>,
    main_worker_data: Option<&mut [usize]>,
    group_number: usize,
    npivots_processed: usize,
) -> usize
where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    let n = args.n;

    // ------------ PHASE 2.1  ------------
    // MAIN_WORKER generates the column-pairs of this group

    if let Some(used) = main_worker_data {
        // SAFETY: There were npivots >= npivots_rotated column-pairs generated in phase 1. So
        // there are enough initialised elements.
        //
        // Also, in phase 2.1, only the MAIN_WORKER is active
        let p = unsafe {
            std::slice::from_raw_parts_mut(
                args.p.as_ptr(),
                args.npivots_rotated - npivots_processed,
            )
        };

        let partition_point = crate::common::partition(p, |&(j, k, _)| {
            // A non-independent rotation is already in this group
            if used[j] == group_number || used[k] == group_number {
                return true;
            }

            // Use this rotation for this group
            used[j] = group_number;
            used[k] = group_number;

            false
        });

        let group_size = p.len() - partition_point;
        args.counter.store(group_size, atomic::Ordering::Relaxed);
    }
    // ------------------------------------

    args.barrier.wait(); // Wait for MAIN_THREAD to the rotation

    // ------------ PHASE 2.2  ------------
    // Generate rotations from the chosen column-pairs in this group (in parallel)

    let group_size = args.counter.load(atomic::Ordering::Relaxed);

    // SAFETY: all worker threads are in phase 2.2. Here b is read-only, so casting the b pointer to
    // a MatrixSlice is ok.
    //
    // The pointer is also valid for n*n matrix (guaranteed by the caller)
    let b_slice = unsafe { std::slice::from_raw_parts(args.b.as_ptr(), n * n) };
    let b = na::MatrixSlice::from_slice_generic(b_slice, args.shape.0, args.shape.1);

    // SAFETY: all worker threads are in phase 2.2. Here p is read-only, so casting the p pointer to
    // a MatrixSlice is ok.
    //
    // The pointer is also valid for npivtos >= npivots_rotated >= npivots_processed + group_size
    // elements
    //
    // Format of p:
    //
    //      |--------------- npivots_rotated ----------------|
    // p = [<unprocessed>, <current group>, <previous groups> ]
    //                     |------.------|  |-------.--------|
    //               group_size --'                 '-- npivots_processed
    let p = unsafe {
        std::slice::from_raw_parts(
            args.p
                .as_ptr()
                .add(args.npivots_rotated - npivots_processed - group_size),
            group_size,
        )
    };

    for i in (worker_id..group_size).step_by(t) {
        let (j, k, d) = p[i];
        // SAFETY: Every i is unique amongst the threads and i < npivots_rotated
        //
        // The caller has guaranteed that q has a capacity of at least npivots_rotated >= group_size
        unsafe {
            args.q
                .as_ptr()
                .add(i)
                .write(JacobiRotation::new(j, k, d, &b))
        }
    }
    // ------------------------------------

    return group_size;
}

#[cfg_attr(feature = "profile", inline(never))]
#[cfg_attr(not(feature = "profile"), inline(always))]
pub(super) unsafe fn jts_group_worker_phase_3<D: na::Dim>(
    worker_id: usize,
    t: usize,
    args: &WorkerArgs<'_, D>,
    group_size: usize,
) where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    let n = args.n;

    for i in (worker_id..group_size).step_by(t) {
        // SAFETY: the main worker makes sure rotations in the queue for a particular group are
        // independent. So, each column will only be referenced once in the threads
        //
        // The slice is also valid as j,k < n and b and v are n*n matrices (guaranteed by the
        // caller)
        let get_col_mut = |col, ptr: *mut Float| unsafe {
            na::VectorSliceMut::from_slice_generic(
                std::slice::from_raw_parts_mut(ptr.add(n * col), n),
                args.shape.0,
                na::Const::<1>,
            )
        };

        // SAFETY: the main worker makes sure that there are group_size number of elements
        //
        // i < group_size
        let rotation = unsafe { args.q.as_ptr().add(i).read() };

        let b_col_j = get_col_mut(rotation.j, args.b.as_ptr());
        let b_col_k = get_col_mut(rotation.k, args.b.as_ptr());

        let v_col_j = get_col_mut(rotation.j, args.v.as_ptr());
        let v_col_k = get_col_mut(rotation.k, args.v.as_ptr());

        rotation.apply_to_columns(b_col_j, b_col_k);
        rotation.apply_to_columns(v_col_j, v_col_k);
    }
}

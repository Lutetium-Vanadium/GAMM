use std::sync::atomic;

use nalgebra as na;

use super::{JacobiRotation, WorkerArgs};
use crate::common::Float;

type MainWorkerData<'a> = (&'a mut [(usize, usize, Float)], Vec<usize>);

#[cfg_attr(feature = "profile", inline(never))]
#[cfg_attr(not(feature = "profile"), inline(always))]
pub(super) unsafe fn jts_group_worker_phase_2<D: na::Dim>(
    _worker_id: usize,
    _t: usize,
    args: &WorkerArgs<'_, D>,
    main_worker_data: Option<&mut MainWorkerData<'_>>,
    group_number: usize,
) -> usize
where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    let n = args.n;

    // SAFETY: all worker threads are in phase 2. Here b is read-only, so casting the b pointer to a
    // MatrixSlice is ok.
    //
    // The pointer is also valid for n*n matrix (guaranteed by the caller)
    let b_slice = unsafe { std::slice::from_raw_parts(args.b.as_ptr(), n * n) };
    let b = na::MatrixSlice::from_slice_generic(b_slice, args.shape.0, args.shape.1);

    // The main worker will first find rotations to be performed for this group and then move on to
    // performing rotations themselves
    if let Some((p, used)) = main_worker_data {
        let mut q_index = 0;

        let partition_point = crate::common::partition(*p, |&(j, k, d)| {
            // A non-independent rotation is already in this group
            if used[j] == group_number || used[k] == group_number {
                return true;
            }

            // Rotation is independent, add it to this group.
            args.rotations_queue
                .push((q_index, (j, k, d)))
                .expect("Queue should have enough capacity");
            used[j] = group_number;
            used[k] = group_number;
            q_index += 1;

            false
        });

        // All rotations from the partition_point onwards (inclusive) have been added to the
        // rotations_queue, and so should be removed from consideration for the next group.
        let _ = p.take_mut(partition_point..);

        args.counter.store(q_index, atomic::Ordering::SeqCst);
    }

    loop {
        // Each thread first tries to take a new rotation to perform. If a rotation is found, the
        // thread will apply it
        if let Some((i, (j, k, d))) = args.rotations_queue.pop() {
            // SAFETY: Every i is unique amongst the threads and i < npivots_rotated
            //
            // The caller has guaranteed that q has a capacity of at least npivots_rotated
            unsafe {
                args.q
                    .as_ptr()
                    .add(i)
                    .write(JacobiRotation::new(j, k, d, &b))
            }
            continue;
        }

        let counter = args.counter.load(atomic::Ordering::SeqCst);
        // If no rotation is found, check if the main worker has signaled that the
        // rotations are complete
        if counter != super::GENERATING {
            break counter;
        }
    }
}

#[cfg_attr(feature = "profile", inline(never))]
#[cfg_attr(not(feature = "profile"), inline(always))]
pub(super) unsafe fn jts_group_worker_phase_3<D: na::Dim>(
    worker_id: usize,
    t: usize,
    args: &WorkerArgs<'_, D>,
    q_len: usize,
) where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    let n = args.n;

    for i in (worker_id..q_len).step_by(t) {
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

        // SAFETY: the main worker makes sure that there are q_len number of elements; i < qlen
        let rotation = unsafe { args.q.as_ptr().add(i).read() };

        let b_col_j = get_col_mut(rotation.j, args.b.as_ptr());
        let b_col_k = get_col_mut(rotation.k, args.b.as_ptr());

        let v_col_j = get_col_mut(rotation.j, args.v.as_ptr());
        let v_col_k = get_col_mut(rotation.k, args.v.as_ptr());

        rotation.apply_to_columns(b_col_j, b_col_k);
        rotation.apply_to_columns(v_col_j, v_col_k);
    }
}

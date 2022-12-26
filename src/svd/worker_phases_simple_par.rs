use nalgebra as na;

use super::{JacobiRotation, WorkerArgs};
use crate::common::Float;

#[cfg_attr(feature = "profile", inline(never))]
#[cfg_attr(not(feature = "profile"), inline(always))]
pub(super) unsafe fn jts_group_worker_phase_2<D: na::Dim>(
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

    // SAFETY: all worker threads are the in phase 2. Here b is read-only, so casting the b pointer
    // to a MatrixSlice is ok.
    //
    // The pointer is also valid for n*n matrix (guaranteed by the caller)
    let b_slice = unsafe { std::slice::from_raw_parts(args.b.as_ptr(), n * n) };
    let b = na::MatrixSlice::from_slice_generic(b_slice, args.shape.0, args.shape.1);

    for i in (worker_id..args.npivots).step_by(t) {
        // SAFETY: Every i is unique amongst the threads.
        //
        // The caller has guaranteed that q has at least npivots elements and p has at least
        // n choose 2 >= npivots elements.
        unsafe {
            let (j, k, d) = args.p.as_ptr().add(i).read();
            args.q
                .as_ptr()
                .add(i)
                .write(JacobiRotation::new(j, k, d, &b))
        }
    }
}

#[cfg_attr(feature = "profile", inline(never))]
#[cfg_attr(not(feature = "profile"), inline(always))]
pub(super) unsafe fn jts_group_worker_phase_3<D: na::Dim>(
    worker_id: usize,
    _t: usize,
    args: &WorkerArgs<'_, D>,
) where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    let n = args.n;

    if worker_id == super::MAIN_WORKER {
        // SAFETY: only the main worker executes this code. So we have exclusive access to all
        // structures here.
        let q = unsafe { std::slice::from_raw_parts(args.q.as_ptr(), args.npivots) };

        let b_slice = unsafe { std::slice::from_raw_parts_mut(args.b.as_ptr(), n * n) };
        let mut b = na::MatrixSliceMut::from_slice_generic(b_slice, args.shape.0, args.shape.1);

        let v_slice = unsafe { std::slice::from_raw_parts_mut(args.v.as_ptr(), n * n) };
        let mut v = na::MatrixSliceMut::from_slice_generic(v_slice, args.shape.0, args.shape.1);

        for &rotation in q {
            rotation.apply_to(&mut b);
            rotation.apply_to(&mut v);
        }
    }
}

use std::{
    ptr::NonNull,
    sync::{
        atomic::{self, AtomicUsize},
        Barrier,
    },
};

#[cfg(feature = "group")]
use crossbeam_queue::ArrayQueue;
use na::RawStorage;
use nalgebra as na;

use crate::common::Float;
use rotation::JacobiRotation;

mod rotation;
mod worker_phase1;
#[cfg(feature = "group")]
mod worker_phases_group;
#[cfg(not(feature = "group"))]
mod worker_phases_simple_par;

use worker_phase1::jts_group_worker_phase_1;
#[cfg(feature = "group")]
use worker_phases_group::{jts_group_worker_phase_2, jts_group_worker_phase_3};
#[cfg(not(feature = "group"))]
use worker_phases_simple_par::{jts_group_worker_phase_2, jts_group_worker_phase_3};

pub struct Svd<D: na::Dim>
where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
{
    pub u: na::OMatrix<Float, D, D>,
    pub sv: na::OVector<Float, D>,
    pub v: na::OMatrix<Float, D, D>,
}

impl<D: na::Dim> Svd<D>
where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    fn sorted_sv(mut self) -> Self {
        const VALUE_PROCESSED: usize = usize::MAX;

        // Collect the singular values with their original index, ...
        let mut singular_values = self.sv.map_with_location(|r, _, e| (e, r));
        assert_ne!(
            singular_values.data.shape().0.value(),
            VALUE_PROCESSED,
            "Too many singular values"
        );

        // ... sort the singular values, ...
        singular_values
            .as_mut_slice()
            .sort_unstable_by(|(a, _), (b, _)| b.partial_cmp(a).expect("Singular value was NaN"));

        // ... and store them.
        self.sv
            .zip_apply(&singular_values, |value, (new_value, _)| {
                value.clone_from(&new_value)
            });

        // Calculate required permutations given the sorted indices.
        // We need to identify all circles to calculate the required swaps.
        let mut permutations =
            na::PermutationSequence::identity_generic(singular_values.data.shape().0);

        for i in 0..singular_values.len() {
            let mut index_1 = i;
            let mut index_2 = singular_values[i].1;

            // Check whether the value was already visited ...
            while index_2 != VALUE_PROCESSED // ... or a "double swap" must be avoided.
                    && singular_values[index_2].1 != VALUE_PROCESSED
            {
                // Add the permutation ...
                permutations.append_permutation(index_1, index_2);
                // ... and mark the value as visited.
                singular_values[index_1].1 = VALUE_PROCESSED;

                index_1 = index_2;
                index_2 = singular_values[index_1].1;
            }
        }

        // Permute the optional components
        permutations.permute_columns(&mut self.u);
        permutations.permute_columns(&mut self.v);

        self
    }

    pub fn jts_seq(
        matrix: na::OMatrix<Float, D, D>,
        tol: Float,
        tau: usize,
        max_sweeps: usize,
    ) -> Self {
        let delta = tol * matrix.norm_squared();
        if cfg!(feature = "print-iter") {
            print!("DELTA: {}", delta);
        }

        let (n_generic, m_generic) = matrix.shape_generic();
        let n = n_generic.value();

        let mut v = na::OMatrix::identity_generic(n_generic, m_generic);
        let mut b = matrix;

        let npivots = n * (n - 1) / 2;
        let npivots_rotated = npivots / tau;

        let mut p = Vec::with_capacity(npivots);
        // Paper uses a queue, but the `q.drain(..)` will give the same effect
        let mut q = Vec::with_capacity(npivots_rotated);

        for i in 0..max_sweeps {
            for j in 0..(n - 1) {
                for k in (j + 1)..n {
                    p.push((j, k, b.column(j).dot(&b.column(k))));
                }
            }

            // Should this sort happen before or after the truncate? The algorithm block indicates
            // after, but that makes less sense (why compute all the pivots, and not just the
            // truncated amount?). The text seems to suggest that the truncate is performed after as
            // well.

            // Sort by descending order of dot-products
            p.sort_unstable_by(|(_, _, a), (_, _, b)| {
                b.partial_cmp(&a).expect("Singular value was NaN")
            });

            p.truncate(npivots_rotated);

            if p[0].2 < delta {
                if cfg!(feature = "print-iter") {
                    println!("\t{} iterations", i);
                }
                break;
            }

            for (j, k, d) in p.drain(..) {
                q.push(JacobiRotation::new(j, k, d, &b));
            }

            for rotation in q.drain(..) {
                rotation.apply_to(&mut b);
                rotation.apply_to(&mut v);
            }
        }
        // Should the matrices be calculated and returned if max_sweeps is reached? Unclear

        let mut sv = na::OVector::zeros_generic(n_generic, na::Const::<1>);

        // Convert U from orthogonal to orthonormal
        for i in 0..n {
            sv[i] = b.column(i).norm();
            if sv[i] < tol {
                sv[i] = 0.0;
                b.column_mut(i).scale_mut(0.0);
            } else {
                b.column_mut(i).unscale_mut(sv[i]);
            }
        }

        Svd { u: b, v, sv }.sorted_sv()
    }

    pub fn jts_par(
        matrix: na::OMatrix<Float, D, D>,
        tol: Float,
        tau: usize,
        max_sweeps: usize,
        t: usize,
    ) -> Self {
        let delta = tol * matrix.norm_squared();
        if cfg!(feature = "print-iter") {
            print!("DELTA: {}", delta);
        }

        let (n_generic, m_generic) = matrix.shape_generic();
        let n = n_generic.value();
        assert_eq!(matrix.shape(), (n, n));

        let mut v = na::OMatrix::identity_generic(n_generic, m_generic);
        let mut b = matrix;

        let npivots = n * (n - 1) / 2;
        let npivots_rotated = npivots / tau;

        let mut p = Vec::with_capacity(npivots);
        // Paper uses a queue, but the `q.drain(..)` will give the same effect
        let mut q = Vec::with_capacity(npivots_rotated);

        let barrier = Barrier::new(t);
        let counter = AtomicUsize::new(0);

        // For a n columns, there is a maximum of n/2 independent rotations that can occur.
        #[cfg(feature = "group")]
        let rotations_queue = ArrayQueue::new(n / 2);

        let worker_args = WorkerArgs {
            n,
            npivots,
            npivots_rotated,
            max_sweeps,
            delta,
            b: NonNull::new(b.as_mut_ptr()).expect("b is not empty"),
            v: NonNull::new(v.as_mut_ptr()).expect("v is not empty"),
            p: NonNull::new(p.as_mut_ptr()).expect("p is allocated"),
            q: NonNull::new(q.as_mut_ptr()).expect("q is allocated"),
            shape: (n_generic, m_generic),
            barrier: &barrier,
            counter: &counter,
            #[cfg(feature = "group")]
            rotations_queue: &rotations_queue,
        };

        std::thread::scope(move |s| {
            let handles: Vec<_> = (0..t)
                .map(|i| {
                    let cloned = worker_args.clone();
                    // SAFETY
                    //
                    // - `npivots = n*(n-1)/2`
                    // - `npivots_rotated <= npivots`
                    // - `b` points to a valid writeable `n*n` matrix buffer
                    // - `v` points to a valid writeable `n*n` matrix buffer
                    // - `p` points to a valid writeable `npivots` `Vec` buffer
                    // - `q` points to a valid writeable `npivots_rotated` `Vec` buffer
                    // - `p_capacity` is the capacity of the `q` `Vec`.
                    // - `shape = (n, n)`
                    // - `barrier` is not waited on by any other threads, only threads running this
                    //   function
                    // - `completed` is not wrote to by any other threads, only threads running this
                    //   function
                    // - `worker_id` is unique to all currently running `jts_group_worker`s and `t`
                    //   is the number of workers
                    s.spawn(move || unsafe { jts_group_worker(i, t, cloned) })
                })
                .collect();

            let _ = handles.into_iter().map(|h| h.join());
        });

        if cfg!(feature = "print-iter") {
            println!("\t{} iterations", counter.into_inner());
        }

        // Should the matrices be calculated and returned if max_sweeps is reached? Unclear

        let mut sv = na::OVector::zeros_generic(n_generic, na::Const::<1>);

        // Convert U from orthogonal to orthonormal
        for i in 0..n {
            sv[i] = b.column(i).norm();
            if sv[i] < tol {
                sv[i] = 0.0;
                b.column_mut(i).scale_mut(0.0);
            } else {
                b.column_mut(i).unscale_mut(sv[i]);
            }
        }

        Svd { u: b, v, sv }.sorted_sv()
    }
}

#[derive(Clone)]
struct WorkerArgs<'b, D> {
    n: usize,
    npivots: usize,
    npivots_rotated: usize,
    max_sweeps: usize,
    delta: Float,
    b: NonNull<Float>,
    v: NonNull<Float>,
    /// (j, k, b_j.dot(b_k))
    p: NonNull<(usize, usize, Float)>,
    q: NonNull<JacobiRotation>,
    shape: (D, D),
    barrier: &'b Barrier,
    counter: &'b AtomicUsize,
    /// (index, element of p)
    #[cfg(feature = "group")]
    rotations_queue: &'b ArrayQueue<(usize, (usize, usize, Float))>,
}

// SAFETY: the pointers are aliased, however, it is only mutably accessed in a non-aliasing way
unsafe impl<'b, D: Send> Send for WorkerArgs<'b, D> {}

const MAIN_WORKER: usize = 0;
const COMPLETED: usize = usize::MAX;
const GENERATING: usize = usize::MAX - 1;

fn assert_copy<T: Copy>() {}
/// # Safety
///
/// The caller must guarantee the following:
/// - `npivots = n*(n-1)/2`
/// - `npivots_rotated <= npivots`
/// - `b` points to a valid writeable `n*n` matrix buffer
/// - `v` points to a valid writeable `n*n` matrix buffer
/// - `p` points to a valid writeable `npivots` `Vec` buffer
/// - `q` points to a valid writeable `npivots_rotated` `Vec` buffer
/// - `p_capacity` is the capacity of the `p` `Vec`.
/// - `shape = (n, n)`
/// - `barrier` is not waited on by any other threads, only threads running this function
/// - `completed` is not wrote to by any other threads, only threads running this function
/// - `worker_id` is unique to all currently running `jts_group_worker`s and `t` is the number of
///   workers
unsafe fn jts_group_worker<D: na::Dim>(worker_id: usize, t: usize, args: WorkerArgs<'_, D>)
where
    na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    na::DefaultAllocator: na::allocator::Allocator<Float, D>,
    na::DefaultAllocator: na::allocator::Allocator<(Float, usize), D>,
    na::DefaultAllocator: na::allocator::Allocator<(usize, usize), D>,
{
    // Make sure there are no changes that make these non-copy. The code below depends on these
    // being copy
    assert_copy::<(usize, usize, Float)>();
    assert_copy::<JacobiRotation>();

    let mut c = args.max_sweeps;

    for i in 0..args.max_sweeps {
        // ============  PHASE 1  ============
        // Generate columns on which rotations will be applied

        unsafe { jts_group_worker_phase_1(worker_id, t, &args) };

        args.barrier.wait(); // Wait for MAIN_WORKER to sort and (possibly) set exit condition

        // Barrier takes care of synchronisation, so relaxed ordering is fine
        if args.counter.load(atomic::Ordering::Relaxed) == COMPLETED {
            c = i;
            break;
        }
        // ===================================

        #[cfg(feature = "group")]
        {
            // SAFETY: There were npivots >= npivots_rotated column-pairs generated in phase 1. So
            // there are enough initialised elements
            let mut main_worker_data = (worker_id == MAIN_WORKER).then(|| unsafe {
                (
                    std::slice::from_raw_parts_mut(args.p.as_ptr(), args.npivots_rotated),
                    // used: Vec -- each element represents column i being used in group used[i]
                    vec![usize::MAX; args.n],
                )
            });

            for group_number in 0.. {
                // pre phase 2 check if main worker set completed last time round
                if args.counter.load(atomic::Ordering::SeqCst) == COMPLETED {
                    break;
                }

                // ============  PHASE 2  ============
                // Generate rotations for the column-pairs in this group (in parallel).
                let q_len = unsafe {
                    jts_group_worker_phase_2(
                        worker_id,
                        t,
                        &args,
                        main_worker_data.as_mut(),
                        group_number,
                    )
                };
                // ===================================

                args.barrier.wait(); // Wait for all threads to finish generating rotations

                // ============  PHASE 3  ============
                // Apply Jacobi rotations in parallel

                unsafe { jts_group_worker_phase_3(worker_id, t, &args, q_len) }

                // ===================================

                // post phase-3 for main worker -- perform before barrier so that
                if let Some((p, _)) = main_worker_data.as_mut() {
                    args.counter.store(
                        if p.is_empty() { COMPLETED } else { GENERATING },
                        atomic::Ordering::SeqCst,
                    );
                }

                args.barrier.wait(); // Wait for all threads to finish this group
            }
        }

        #[cfg(not(feature = "group"))]
        {
            // ============  PHASE 2  ============
            // Generate rotations based on the top 1/tau fraction of column-pairs (in parallel).
            unsafe { jts_group_worker_phase_2(worker_id, t, &args) };
            // ===================================

            args.barrier.wait(); // Wait for all threads to finish generating the Jacobi rotations.

            // ============  PHASE 3  ============
            // Apply Jacobi rotations
            // Non parallel -- only MAIN_WORKER applies all rotations
            unsafe { jts_group_worker_phase_3(worker_id, t, &args) };
            // ===================================
        }

        args.barrier.wait(); // Wait for iteration to complete so next iteration can be started
    }

    if cfg!(feature = "print-iter") {
        args.barrier.wait(); // Wait for all threads to have read counter == COMPLETED

        if worker_id == MAIN_WORKER {
            args.counter.store(c, atomic::Ordering::Relaxed);
        }
    }
}

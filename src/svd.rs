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
use na::SimdComplexField;
use nalgebra as na;

use crate::common::Float;

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
            #[cfg(feature = "group")]
            p_capacity: p.capacity(),
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
    #[cfg(feature = "group")]
    p_capacity: usize,
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

    let n = args.n;
    let mut c = args.max_sweeps;

    for i in 0..args.max_sweeps {
        // ============  PHASE 1  ============
        // Generate columns on which rotations will be applied

        // ------------ PHASE 1.1 ------------
        // Generate all the possible column pairs (in parallel).
        {
            // SAFETY: all worker threads are the in phase 1.1. Here b is read-only, so casting the
            // b pointer to a MatrixSlice is ok.
            //
            // The pointer is also valid for n*n matrix (guaranteed by the caller)
            let b_slice = unsafe { std::slice::from_raw_parts(args.b.as_ptr(), n * n) };
            let b = na::MatrixSlice::from_slice_generic(b_slice, args.shape.0, args.shape.1);
            for j in (worker_id..(n - 1)).step_by(t) {
                for k in (j + 1)..n {
                    // SAFETY: caller guarantees p points to a pre-allocated buffer which is
                    // writeable. Each index here is unique and so writing to it in parallel is ok.
                    unsafe {
                        args.p
                            .as_ptr()
                            // Each iteration (j) has (n-(j+1)) inner iterations. Current index will
                            // be sum of number of previous inner iterations plus number of inner
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

        // Should this sort happen before or after the truncate? The algorithm block indicates
        // after, but that makes less sense (why compute all the pivots, and not just the
        // truncated amount?). The text seems to suggest that the truncate is performed after as
        // well.

        args.barrier.wait(); // Wait for p to be filled by all threads. Only then commence sorting

        // ------------ PHASE 1.2 ------------
        // Sort the column-pairs by dot-product. This is performed only by the main worker.

        if worker_id == MAIN_WORKER {
            // SAFETY: caller guarantees p points to a pre-allocated buffer which is npivots long.
            // The values of p were written already in phase 1.1
            let p = unsafe { std::slice::from_raw_parts_mut(args.p.as_ptr(), args.npivots) };

            // Sort by descending order of dot-products
            p.sort_unstable_by(|(_, _, a), (_, _, b)| {
                b.partial_cmp(&a).expect("Singular value was NaN")
            });

            // Barrier takes care of synchronisation, so relaxed ordering is fine
            args.counter.store(
                if p[0].2 < args.delta {
                    COMPLETED
                } else {
                    GENERATING
                },
                atomic::Ordering::Relaxed,
            );
        }

        args.barrier.wait(); // Wait for MAIN_WORKER to sort and (possibly) set exit condition

        // Barrier takes care of synchronisation, so relaxed ordering is fine
        if args.counter.load(atomic::Ordering::Relaxed) == COMPLETED {
            c = i;
            break;
        }
        // ===================================

        #[cfg(feature = "group")]
        {
            // SAFETY: The caller guarantees that the p pointer and p_capacity is accurate for
            // the Vec. There are also npivots_rotated column-pairs created in phase 1.
            //
            // Since the Vec is in an Undroppable, it will not be dropped. It will also not
            // reallocate as no resizing functions are called.
            let mut main_worker_data = (worker_id == MAIN_WORKER).then(|| unsafe {
                (
                    undroppable::Undroppable::new(Vec::from_raw_parts(
                        args.p.as_ptr(),
                        args.npivots_rotated,
                        args.p_capacity,
                    )),
                    // used: Vec -- each element represents column i being used in group used[i]
                    vec![usize::MAX; n],
                )
            });

            for group_number in 0.. {
                // pre phase 2 check if main worker set completed last time round
                if args.counter.load(atomic::Ordering::SeqCst) == COMPLETED {
                    break;
                }

                // ============  PHASE 2  ============
                // Generate rotations for the column-pairs in this group (in parallel).
                let q_len = {
                    // SAFETY: all worker threads are in phase 2. Here b is read-only, so casting
                    // the b pointer to a MatrixSlice is ok.
                    //
                    // The pointer is also valid for n*n matrix (guaranteed by the caller)
                    let b_slice = unsafe { std::slice::from_raw_parts(args.b.as_ptr(), n * n) };
                    let b =
                        na::MatrixSlice::from_slice_generic(b_slice, args.shape.0, args.shape.1);

                    // The main worker will first find rotations to be performed for this group and then
                    // move on to performing rotations themselves
                    if let Some((p, used)) = main_worker_data.as_mut() {
                        let mut q_index = 0;
                        p.retain(|&(j, k, d)| {
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

                        args.counter.store(q_index, atomic::Ordering::SeqCst);
                    }

                    loop {
                        // Each thread first tries to take a new rotation to perform. If a rotation is
                        // found, the thread will apply it
                        if let Some((i, (j, k, d))) = args.rotations_queue.pop() {
                            // SAFETY: Every i is unique amongst the threads and i < npivots_rotated
                            //
                            // The caller has guaranteed that q has a capacity of at least
                            // npivots_rotated
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
                        if counter != GENERATING {
                            break counter;
                        }
                    }
                };
                // ===================================

                args.barrier.wait(); // Wait for all threads to finish generating rotations

                // ============  PHASE 3  ============
                // Apply Jacobi rotations in parallel

                for i in (worker_id..q_len).step_by(t) {
                    // SAFETY: the main worker makes sure rotations in the queue for a particular
                    // group are independent. So, each column will only be referenced once in
                    // the threads
                    //
                    // The slice is also valid as j,k < n and b and v are n*n matrices
                    // (guaranteed by the caller)
                    let get_col_mut = |col, ptr: *mut Float| unsafe {
                        na::VectorSliceMut::from_slice_generic(
                            std::slice::from_raw_parts_mut(ptr.add(n * col), n),
                            args.shape.0,
                            na::Const::<1>,
                        )
                    };

                    // SAFETY: the main worker makes sure that there are q_len number of elements
                    //
                    // i < qlen
                    let rotation = unsafe { args.q.as_ptr().add(i).read() };

                    let b_col_j = get_col_mut(rotation.j, args.b.as_ptr());
                    let b_col_k = get_col_mut(rotation.k, args.b.as_ptr());

                    let v_col_j = get_col_mut(rotation.j, args.v.as_ptr());
                    let v_col_k = get_col_mut(rotation.k, args.v.as_ptr());

                    rotation.apply_to_columns(b_col_j, b_col_k);
                    rotation.apply_to_columns(v_col_j, v_col_k);
                }

                // ===================================

                // post phase-3 for main worker -- perform before barrier so that
                if let Some((p, _)) = main_worker_data.as_mut() {
                    args.counter.store(
                        if p.is_empty() { COMPLETED } else { GENERATING },
                        atomic::Ordering::SeqCst,
                    );

                    // sanity check -- we do not actually own the q Vec, so nothing about should
                    // change
                    assert_eq!(p.as_mut_ptr(), args.p.as_ptr());
                    assert_eq!(p.capacity(), args.p_capacity);
                }

                args.barrier.wait(); // Wait for all threads to finish this group
            }
        }

        #[cfg(not(feature = "group"))]
        {
            // ============  PHASE 2  ============
            // Generate rotations based on the top 1/tau fraction of column-pairs (in parallel).
            {
                // SAFETY: all worker threads are the in phase 2. Here b is read-only, so casting
                // the b pointer to a MatrixSlice is ok.
                //
                // The pointer is also valid for n*n matrix (guaranteed by the caller)
                let b_slice = unsafe { std::slice::from_raw_parts(args.b.as_ptr(), n * n) };
                let b = na::MatrixSlice::from_slice_generic(b_slice, args.shape.0, args.shape.1);

                for i in (worker_id..args.npivots_rotated).step_by(t) {
                    // SAFETY: Every i is unique amongst the threads.
                    //
                    // The caller has guaranteed that q has at least npivots_rotated elements and p
                    // has at least npivots >= npivots_rotated elements.
                    unsafe {
                        let (j, k, d) = args.p.as_ptr().add(i).read();
                        args.q
                            .as_ptr()
                            .add(i)
                            .write(JacobiRotation::new(j, k, d, &b))
                    }
                }
            }
            // ===================================

            args.barrier.wait(); // Wait for all threads to finish generating the Jacobi rotations.

            // ============  PHASE 3  ============
            // Apply Jacobi rotations
            // Non parallel -- only MAIN_WORKER applies all rotations

            if worker_id == MAIN_WORKER {
                // SAFETY: only the main worker executes this code. So we have exclusive access to
                // all structures here.
                let q =
                    unsafe { std::slice::from_raw_parts(args.q.as_ptr(), args.npivots_rotated) };

                let b_slice = unsafe { std::slice::from_raw_parts_mut(args.b.as_ptr(), n * n) };
                let mut b =
                    na::MatrixSliceMut::from_slice_generic(b_slice, args.shape.0, args.shape.1);

                let v_slice = unsafe { std::slice::from_raw_parts_mut(args.v.as_ptr(), n * n) };
                let mut v =
                    na::MatrixSliceMut::from_slice_generic(v_slice, args.shape.0, args.shape.1);

                for &rotation in q {
                    rotation.apply_to(&mut b);
                    rotation.apply_to(&mut v);
                }
            }
        }

        // ===================================

        args.barrier.wait(); // Wait for iteration to complete so next iteration can be started
    }

    args.barrier.wait();

    if worker_id == MAIN_WORKER {
        args.counter.store(c, atomic::Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
struct JacobiRotation {
    c: Float,
    s: Float,
    inv_c: Float,
    t: Float,
    j: usize,
    k: usize,
}

impl JacobiRotation {
    fn new<D: na::Dim, S: na::Storage<Float, D, D>>(
        j: usize,
        k: usize,
        d: Float,
        a: &na::Matrix<Float, D, D, S>,
    ) -> Self
    where
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    {
        let gamma = (a.column(k).norm_squared() - a.column(j).norm_squared()) / (2.0 * d);
        let t = (1.0 / (gamma.simd_abs() + (gamma.simd_powi(2) + 1.0).simd_sqrt())).copysign(gamma);
        let inv_c = (t.simd_powi(2) + 1.0).simd_sqrt();
        let c = 1.0 / inv_c;
        let s = t * c;
        Self {
            c,
            s,
            j,
            k,
            inv_c,
            t,
        }
    }

    fn apply_to<D: na::Dim, S>(self, mat: &mut na::Matrix<Float, D, D, S>)
    where
        S: na::StorageMut<Float, D, D>,
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    {
        let (mut a, mut b) = mat.columns_range_pair_mut(0..self.k, self.k..);

        self.apply_to_columns(a.column_mut(self.j), b.column_mut(0));
    }

    fn apply_to_columns<D: na::Dim, S: na::StorageMut<Float, D, na::U1>>(
        self,
        mut col_j: na::Matrix<Float, D, na::U1, S>,
        mut col_k: na::Matrix<Float, D, na::U1, S>,
    ) where
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    {
        // BJ_j = cB_j - sB_k
        col_j.axpy(-self.s, &col_k, self.c);

        // BJ_k = cB_k + sB_j
        //
        // Since we have modified B_j in place, now B_j = BJ_j = cB_j - sB_k.
        // So B_j = (BJ_j + sB_k)/c
        // So,
        // BJ_k = cB_k + s(BJ_j + sB_k)/c
        //      = ((c^2+s^2)B_k + tcBJ_j)/c
        // BJ_k = (1/c)B_k + tBJ_j
        col_k.axpy(self.t, &col_j, self.inv_c);
    }
}

#[cfg(feature = "group")]
mod undroppable {
    use std::{
        mem::ManuallyDrop,
        ops::{Deref, DerefMut},
    };

    #[repr(transparent)]
    pub(super) struct Undroppable<T>(ManuallyDrop<T>);

    impl<T> Undroppable<T> {
        pub(super) fn new(value: T) -> Self {
            Self(ManuallyDrop::new(value))
        }
    }

    impl<T> Deref for Undroppable<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &*self.0
        }
    }

    impl<T> DerefMut for Undroppable<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut *self.0
        }
    }
}

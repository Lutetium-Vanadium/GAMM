use nalgebra as na;
use num_traits::Zero;

pub type Float = f32;

#[derive(Default)]
pub struct ZeroedColumns {
    head: usize,
    next_zeroed: Vec<usize>,
    nzeroed: usize,
}

impl ZeroedColumns {
    const NON_ZERO: usize = usize::MAX;

    pub fn new_no_zeroed(size: usize) -> Self {
        Self {
            next_zeroed: vec![Self::NON_ZERO; size],
            head: 0,
            nzeroed: 0,
        }
    }

    pub fn new_all_zeroed(size: usize) -> Self {
        Self {
            next_zeroed: (1..=size).collect(),
            head: 0,
            nzeroed: size,
        }
    }

    pub fn new_from_matrix<T: Zero, R: na::Dim, C: na::Dim, S: na::Storage<T, R, C>>(
        matrix: &na::Matrix<T, R, C, S>,
    ) -> Self {
        let n = matrix.ncols();
        let mut this = Self::new_no_zeroed(n);

        for (i, c) in matrix.column_iter().enumerate() {
            if c.iter().all(|x| x.is_zero()) {
                this.set_zeroed(i);
            }
        }

        this
    }

    pub fn nzeroed(&self) -> usize {
        self.nzeroed
    }

    pub fn set_zeroed(&mut self, index: usize) {
        assert_eq!(self.next_zeroed[index], Self::NON_ZERO);
        self.next_zeroed[index] = self.head;
        self.head = index;
        self.nzeroed += 1;
    }

    pub fn get_next_zeroed(&mut self) -> usize {
        assert_ne!(self.nzeroed, 0);

        let old_head = self.head;
        self.head = self.next_zeroed[self.head];
        self.next_zeroed[old_head] = Self::NON_ZERO;
        self.nzeroed -= 1;
        old_head
    }

    pub fn check_matching_zeroed<R: na::Dim, C: na::Dim, S>(&self, m: &na::Matrix<Float, R, C, S>)
    where
        S: na::Storage<Float, R, C>,
    {
        assert_eq!(m.ncols(), self.next_zeroed.len());

        for (i, c) in m.column_iter().enumerate() {
            assert_eq!(
                c.iter().all(|x| x.is_zero()),
                self.next_zeroed[i] != Self::NON_ZERO,
                "column {}",
                i
            );
        }
    }
}

#[inline(always)]
pub fn qr<R: na::Dim, C: na::Dim, SA, SB>(
    mut q: na::Matrix<Float, R, C, SA>,
    mut r: na::Matrix<Float, C, C, SB>,
    r_transposed: bool,
) -> (na::Matrix<Float, R, C, SA>, na::Matrix<Float, C, C, SB>)
where
    SA: na::Storage<Float, R, C> + na::RawStorageMut<Float, R, C>,
    SB: na::Storage<Float, C, C> + na::RawStorageMut<Float, C, C>,
{
    let (_n, m) = q.shape();
    assert_eq!(r.shape(), (m, m));

    for k in 0..m {
        for i in 0..k {
            let (mut q_k, q_i) = q.columns_range_pair_mut(k, i);
            let r_i = if r_transposed { (k, i) } else { (i, k) };
            r[r_i] = q_i.dot(&q_k);
            q_k.axpy(-r[r_i], &q_i, 1.0)
        }

        r[(k, k)] = q.column(k).norm();
        q.column_mut(k).unscale_mut(r[(k, k)]);
    }

    (q, r)
}

pub fn find_l2_norm(m: na::DMatrix<Float>) -> Float {
    m.svd(false, false).singular_values[0]
}

pub fn hardware_concurrency() -> usize {
    std::env::var("HARDWARE_CONCURRENCY")
        .map(|s| s.parse::<usize>().ok())
        .ok()
        .flatten()
        .or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .ok()
        })
        .unwrap_or(1)
}

/// Partition the slice into two segments according to predicate. Elements for which the predicate
/// returns `true` are kept in the beginning in the same order as which they occur in the slice.
/// Elements for which the predicate returns `false` are stored starting at the returned index in
/// random order.
pub fn partition<T, P: FnMut(&T) -> bool>(slice: &mut [T], mut predicate: P) -> usize {
    let mut parition_point = 0;

    for i in 0..slice.len() {
        if predicate(&slice[i]) {
            slice.swap(parition_point, i);
            parition_point += 1;
        }
    }

    parition_point
}

use nalgebra as na;
use num_traits::Zero;

pub type Float = f32;

// pub const DIM_M1: usize = 200;
// pub const DIM_M2: usize = 100;
// pub const DIM_D: usize = 1000;
// pub const L: usize = 100;

pub const DIM_M1: usize = 1000;
pub const DIM_M2: usize = 1000;
pub const DIM_D: usize = 10000;
pub const L: usize = 400;

pub const BETA: Float = 28.0;

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

    pub fn nzeroed(&self) -> usize {
        self.nzeroed
    }

    pub fn set_zeroed(&mut self, index: usize) {
        debug_assert_eq!(self.next_zeroed[index], Self::NON_ZERO);
        self.next_zeroed[index] = self.head;
        self.head = index;
        self.nzeroed += 1;
    }

    pub fn get_next_zeroed(&mut self) -> usize {
        debug_assert_ne!(self.nzeroed, 0);

        let old_head = self.head;
        self.head = self.next_zeroed[self.head];
        self.next_zeroed[old_head] = Self::NON_ZERO;
        self.nzeroed -= 1;
        old_head
    }

    pub fn check_matching_zeroed(&self, m: &na::DMatrix<Float>) {
        assert_eq!(m.shape().1, self.next_zeroed.len());

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

use nalgebra as na;

use crate::common::Float;

pub trait SVDCalculator {
    fn svd<D>(&self, matrix: na::OMatrix<Float, D, D>) -> super::Svd<D>
    where
        D: na::Dim,
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>
            + na::allocator::Allocator<Float, D>
            + na::allocator::Allocator<(Float, usize), D>
            + na::allocator::Allocator<(usize, usize), D>;
}

pub struct SeqJTSConfig {
    pub tol: Float,
    pub tau: usize,
    pub max_sweeps: usize,
}

impl Default for SeqJTSConfig {
    fn default() -> Self {
        Self {
            tol: super::TOL,
            tau: super::TAU,
            max_sweeps: super::MAX_SWEEPS,
        }
    }
}

impl SVDCalculator for SeqJTSConfig {
    fn svd<D>(&self, matrix: na::OMatrix<Float, D, D>) -> super::Svd<D>
    where
        D: na::Dim,
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>
            + na::allocator::Allocator<Float, D>
            + na::allocator::Allocator<(Float, usize), D>
            + na::allocator::Allocator<(usize, usize), D>,
    {
        super::Svd::jts_seq(matrix, self.tol, self.tau, self.max_sweeps)
    }
}

pub struct ParJTSConfig {
    pub tol: Float,
    pub tau: usize,
    pub max_sweeps: usize,
    pub t: usize,
}

impl ParJTSConfig {
    pub fn new(t: usize) -> Self {
        Self {
            tol: super::TOL,
            tau: super::TAU,
            max_sweeps: super::MAX_SWEEPS,
            t,
        }
    }

    pub fn with_t(mut self, t: usize) -> Self {
        self.t = t;
        self
    }
}

impl SVDCalculator for ParJTSConfig {
    fn svd<D>(&self, matrix: na::OMatrix<Float, D, D>) -> super::Svd<D>
    where
        D: na::Dim,
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>
            + na::allocator::Allocator<Float, D>
            + na::allocator::Allocator<(Float, usize), D>
            + na::allocator::Allocator<(usize, usize), D>,
    {
        super::Svd::jts_par(matrix, self.tol, self.tau, self.max_sweeps, self.t)
    }
}

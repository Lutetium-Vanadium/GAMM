use nalgebra as na;

use crate::common::Float;
use scoped_pool::Pool;

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

pub struct ParJTSConfig <'a> {
    pub tol: Float,
    pub tau: usize,
    pub max_sweeps: usize,
    pub pool: &'a Pool,
}

impl<'a> ParJTSConfig<'a> {
    pub fn new(pool: &'a Pool) -> Self {
        Self {
            tol: super::TOL,
            tau: super::TAU,
            max_sweeps: super::MAX_SWEEPS,
            pool,
        }
    }

    pub fn with_pool(mut self, pool: &'a Pool) -> Self {
        self.pool = pool;
        self
    }
}

impl<'a> SVDCalculator for ParJTSConfig<'a> {
    fn svd<D>(&self, matrix: na::OMatrix<Float, D, D>) -> super::Svd<D>
    where
        D: na::Dim,
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>
            + na::allocator::Allocator<Float, D>
            + na::allocator::Allocator<(Float, usize), D>
            + na::allocator::Allocator<(usize, usize), D>,
    {
        super::Svd::jts_par(matrix, self.tol, self.tau, self.max_sweeps, self.pool)
    }
}

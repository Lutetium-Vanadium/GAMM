use na::SimdComplexField;
use nalgebra as na;

use crate::common::Float;

#[derive(Debug, Clone, Copy)]
pub struct JacobiRotation {
    pub c: Float,
    pub s: Float,
    pub inv_c: Float,
    pub t: Float,
    pub j: usize,
    pub k: usize,
}

impl JacobiRotation {
    pub fn new<D: na::Dim, S: na::Storage<Float, D, D>>(
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

    pub fn apply_to<D: na::Dim, S>(self, mat: &mut na::Matrix<Float, D, D, S>)
    where
        S: na::StorageMut<Float, D, D>,
        na::DefaultAllocator: na::allocator::Allocator<Float, D, D>,
    {
        let (mut a, mut b) = mat.columns_range_pair_mut(0..self.k, self.k..);

        self.apply_to_columns(a.column_mut(self.j), b.column_mut(0));
    }

    pub fn apply_to_columns<D: na::Dim, S: na::StorageMut<Float, D, na::U1>>(
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

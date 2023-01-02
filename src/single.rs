use nalgebra as na;

use crate::{bamm, common::Float, config, svd};

pub fn beta_coocurring_amm(
    x: &na::DMatrix<Float>,
    y: &na::DMatrix<Float>,
    config: &config::Config,
) -> na::DMatrix<Float> {
    let l = config.l;

    // Start by inserting first l columns.
    //
    // The loop essentially inserts replaces zero columns with columns from x and y until there are
    // no zero columns left, so the first l columns can be copied beforehand
    let mut bx = na::DMatrix::from(x.columns(0, l));
    let mut by = na::DMatrix::from(y.columns(0, l));

    bamm::beta_coocurring_reduction(
        x.columns_range(l..),
        y.columns_range(l..),
        &mut bx,
        &mut by,
        &svd::SeqJTSConfig::default(),
        &config.into(),
    );

    bx * by.transpose()
}

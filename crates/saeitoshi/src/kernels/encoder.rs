//! Encoder matmul entry point. Dispatches through a [`super::Backend`].
//!
//! Two parallelization patterns coexist, selected by `enc.layout`:
//!
//! - **Row-major weights** (legacy backends): rayon splits the batch
//!   dimension — each task processes one batch row, calling the backend
//!   with `batch = 1`. Simple, but every task streams the full `W_enc`
//!   from DRAM, which is the bottleneck the perf-v2 redesign closes.
//! - **Packed weights** (tiled GEMM backends): the tiled kernel does its
//!   own M-block parallelization internally (each task owns a slice of
//!   features; threads share the x panel via L3). This layer calls the
//!   backend exactly once with the full batch.
//!
//! `SAEITOSHI_NO_PARALLEL=1` forces sequential mode in both paths (used
//! for SIMD parity tests so we don't have cross-thread float noise to
//! chase — though for the tiled backend each thread writes disjoint
//! outputs, so cross-thread FP non-determinism isn't actually possible).

use std::sync::OnceLock;

use rayon::prelude::*;

use crate::sae::{EncoderWeights, WeightLayout};

use super::Backend;

pub(crate) fn parallel_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("SAEITOSHI_NO_PARALLEL").is_err())
}

#[inline]
pub fn encode_f32(
    backend: &Backend,
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts: &mut [f32],
    batch: usize,
) {
    match enc.layout {
        WeightLayout::PackedPanels { .. } => {
            // The tiled backend handles its own M-block rayon split.
            // Calling once with the full batch keeps a single coherent
            // packed-W stream visible to every thread.
            (backend.encode_f32)(x, enc, pre_acts, batch);
        }
        WeightLayout::RowMajor => {
            if batch <= 1 || !parallel_enabled() {
                (backend.encode_f32)(x, enc, pre_acts, batch);
                return;
            }
            let d_in = enc.d_in;
            let d_sae = enc.d_sae;
            pre_acts
                .par_chunks_exact_mut(d_sae)
                .enumerate()
                .for_each(|(b, pre_row)| {
                    let x_row = &x[b * d_in..(b + 1) * d_in];
                    (backend.encode_f32)(x_row, enc, pre_row, 1);
                });
        }
    }
}

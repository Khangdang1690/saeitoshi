//! Encoder matmul entry point. Dispatches through a [`super::Backend`].
//!
//! For multi-row batches the dispatch layer parallelizes across batch rows
//! with rayon — each thread runs the backend kernel for one batch row.
//! `SAEITOSHI_NO_PARALLEL=1` forces sequential mode (used for the SIMD
//! parity tests so we don't have cross-thread float noise to chase).

use std::sync::OnceLock;

use rayon::prelude::*;

use crate::sae::EncoderWeights;

use super::Backend;

fn parallel_enabled() -> bool {
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

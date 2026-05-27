//! Encoder matmul entry point. Dispatches through a [`super::Backend`].
//!
//! The tiled GEMM backends own their own M-block rayon split: each task
//! covers a slice of `d_sae` features; threads share the input `x` panel
//! via L3 rather than re-streaming the weight matrix.
//!
//! `SAEITOSHI_NO_PARALLEL=1` forces sequential mode inside the tiled
//! backends — used by parity tests so per-thread float reductions stay
//! deterministic.

use std::sync::OnceLock;

use crate::sae::EncoderWeights;

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
    (backend.encode_f32)(x, enc, pre_acts, batch);
}

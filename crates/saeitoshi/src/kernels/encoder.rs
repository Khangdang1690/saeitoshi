//! Encoder matmul entry point. Dispatches through a [`super::Backend`].

use crate::sae::EncoderWeights;

use super::Backend;

#[inline]
pub fn encode_f32(
    backend: &Backend,
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts: &mut [f32],
    batch: usize,
) {
    (backend.encode_f32)(x, enc, pre_acts, batch)
}

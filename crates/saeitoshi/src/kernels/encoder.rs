//! Encoder matmul entry point. Dispatches to a backend (scalar in M1, SIMD
//! variants in M3).

use crate::sae::EncoderWeights;

#[inline]
pub fn encode_f32(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    super::scalar::encode_f32(x, enc, pre_acts, batch);
}

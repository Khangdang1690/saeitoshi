//! aarch64 NEON kernels.
//!
//! Written without ARM hardware in the loop — verified against the
//! [`super::scalar`] reference in CI on macOS-ARM / Linux-aarch64 in M6.

#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::Backend;
use crate::sae::EncoderWeights;

fn encode_f32_neon(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    // SAFETY: NEON is guaranteed on any aarch64 target supported by std,
    // and `select_backend` double-checks via `is_aarch64_feature_detected!`.
    unsafe { encode_f32_neon_inner(x, enc, pre_acts, batch) }
}

#[target_feature(enable = "neon")]
#[allow(clippy::needless_range_loop)]
unsafe fn encode_f32_neon_inner(
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts: &mut [f32],
    batch: usize,
) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;

    for f in 0..d_sae {
        let w_row = &enc.w_enc[f * d_in..(f + 1) * d_in];
        let bias = enc.b_enc[f];
        for b in 0..batch {
            let x_row = &x[b * d_in..(b + 1) * d_in];

            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= d_in {
                let xv = vld1q_f32(x_row.as_ptr().add(i));
                let wv = vld1q_f32(w_row.as_ptr().add(i));
                acc = vfmaq_f32(acc, xv, wv);
                i += 4;
            }

            let mut sum = vaddvq_f32(acc);
            while i < d_in {
                sum += x_row[i] * w_row[i];
                i += 1;
            }
            pre_acts[b * d_sae + f] = sum + bias;
        }
    }
}

pub static NEON: Backend = Backend {
    name: "neon",
    encode_f32: encode_f32_neon,
};

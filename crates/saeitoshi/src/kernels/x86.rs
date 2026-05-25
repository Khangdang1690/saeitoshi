//! x86_64 SIMD kernels: AVX2 + AVX-512.
//!
//! All `#[target_feature]`-gated `unsafe fn`s are wrapped in safe `fn`s
//! that the [`super::Backend`] vtable holds. Safety obligation lives at the
//! dispatch site in [`super::select_backend`].

#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::Backend;
use crate::sae::EncoderWeights;

// ---------------- AVX2 + FMA ----------------

/// Safe wrapper. Stored in [`AVX2`]; called only when AVX2 + FMA are detected.
fn encode_f32_avx2(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    // SAFETY: AVX2 + FMA detection happened in `select_backend` before this
    // pointer was returned. The function-pointer field is private to the
    // crate; there is no public way to call this without the feature check.
    unsafe { encode_f32_avx2_inner(x, enc, pre_acts, batch) }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)]
unsafe fn encode_f32_avx2_inner(
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts: &mut [f32],
    batch: usize,
) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    debug_assert_eq!(x.len(), batch * d_in);
    debug_assert_eq!(pre_acts.len(), batch * d_sae);

    for b in 0..batch {
        let x_row = &x[b * d_in..(b + 1) * d_in];
        let pre_row = &mut pre_acts[b * d_sae..(b + 1) * d_sae];
        for f in 0..d_sae {
            let w_row = &enc.w_enc[f * d_in..(f + 1) * d_in];

            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            // 8 floats per AVX2 register.
            while i + 8 <= d_in {
                let xv = _mm256_loadu_ps(x_row.as_ptr().add(i));
                let wv = _mm256_loadu_ps(w_row.as_ptr().add(i));
                acc = _mm256_fmadd_ps(xv, wv, acc);
                i += 8;
            }

            // Horizontal sum, then add the bias + scalar tail.
            let mut sum = horizontal_sum_avx2(acc);
            while i < d_in {
                sum += x_row[i] * w_row[i];
                i += 1;
            }
            pre_row[f] = sum + enc.b_enc[f];
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Store and sum in order. Deterministic across runs, plays nice with
    // the parity gate (scalar uses left-to-right; this gives the same
    // order over the 8 SIMD lanes).
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), v);
    buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
}

pub static AVX2: Backend = Backend {
    name: "avx2",
    encode_f32: encode_f32_avx2,
};

// ---------------- AVX-512 ----------------

fn encode_f32_avx512(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    // SAFETY: AVX-512F + AVX-512BW detected before backend selection.
    unsafe { encode_f32_avx512_inner(x, enc, pre_acts, batch) }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::needless_range_loop)]
unsafe fn encode_f32_avx512_inner(
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts: &mut [f32],
    batch: usize,
) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    debug_assert_eq!(x.len(), batch * d_in);
    debug_assert_eq!(pre_acts.len(), batch * d_sae);

    for b in 0..batch {
        let x_row = &x[b * d_in..(b + 1) * d_in];
        let pre_row = &mut pre_acts[b * d_sae..(b + 1) * d_sae];
        for f in 0..d_sae {
            let w_row = &enc.w_enc[f * d_in..(f + 1) * d_in];

            let mut acc = _mm512_setzero_ps();
            let mut i = 0;
            // 16 floats per AVX-512 register.
            while i + 16 <= d_in {
                let xv = _mm512_loadu_ps(x_row.as_ptr().add(i));
                let wv = _mm512_loadu_ps(w_row.as_ptr().add(i));
                acc = _mm512_fmadd_ps(xv, wv, acc);
                i += 16;
            }

            let mut sum = _mm512_reduce_add_ps(acc);
            while i < d_in {
                sum += x_row[i] * w_row[i];
                i += 1;
            }
            pre_row[f] = sum + enc.b_enc[f];
        }
    }
}

pub static AVX512: Backend = Backend {
    name: "avx512",
    encode_f32: encode_f32_avx512,
};

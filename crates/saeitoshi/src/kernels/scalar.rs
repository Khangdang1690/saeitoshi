//! Portable scalar reference kernels — the parity gate.
//!
//! These match PyTorch's FP32 op order: row-major, f32 accumulators, no FMA
//! reassociation. SIMD backends (M3) must agree with these to within 1e-5,
//! which transitively guarantees ≤1e-5 vs SAELens.

use crate::sae::{DecoderWeights, EncoderWeights, SparseOut};
use crate::sparsify::TopKScratch;

/// Encoder: `pre_acts[b, f] = b_enc[f] + sum_i x[b, i] * W_enc[f, i]`.
///
/// `x` is `[batch * d_in]`, `pre_acts` is `[batch * d_sae]`, both row-major.
/// `W_enc` is stored as `[d_sae, d_in]` row-major so the inner i-loop strides
/// W_enc contiguously per output feature f.
// Hot kernel — keep integer-indexed loops so the SIMD replacement in M3
// can drop in with the same shape. Iterator chains here hurt readability
// and don't help the compiler vectorize the scalar path.
#[allow(clippy::needless_range_loop)]
pub fn encode_f32(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    debug_assert_eq!(x.len(), batch * d_in);
    debug_assert_eq!(pre_acts.len(), batch * d_sae);
    debug_assert_eq!(enc.w_enc.len(), d_sae * d_in);
    debug_assert_eq!(enc.b_enc.len(), d_sae);

    for b in 0..batch {
        let x_row = &x[b * d_in..(b + 1) * d_in];
        let pre_row = &mut pre_acts[b * d_sae..(b + 1) * d_sae];
        for f in 0..d_sae {
            let w_row = &enc.w_enc[f * d_in..(f + 1) * d_in];
            let mut acc = enc.b_enc[f];
            for i in 0..d_in {
                acc += x_row[i] * w_row[i];
            }
            pre_row[f] = acc;
        }
    }
}

/// Sparse decoder: `out[b, :] = b_dec + sum_{(f, v) in z[b]} v * W_dec[f, :]`.
pub fn decode_sparse(z: &SparseOut, dec: &DecoderWeights, out: &mut [f32]) {
    let d_in = dec.d_in;
    let batch = z.batch_size();
    debug_assert_eq!(out.len(), batch * d_in);
    debug_assert_eq!(dec.w_dec.len(), dec.d_sae * d_in);
    debug_assert_eq!(dec.b_dec.len(), d_in);

    for b in 0..batch {
        let out_row = &mut out[b * d_in..(b + 1) * d_in];
        out_row.copy_from_slice(&dec.b_dec);
        let s = z.row_offsets[b] as usize;
        let e = z.row_offsets[b + 1] as usize;
        for k in s..e {
            let f = z.indices[k] as usize;
            let v = z.values[k];
            let w_row = &dec.w_dec[f * d_in..(f + 1) * d_in];
            for j in 0..d_in {
                out_row[j] += v * w_row[j];
            }
        }
    }
}

/// Select the k largest entries from `scores` and write them, sorted by
/// index ascending, into `scratch.indexed`. Ties broken by index ascending.
///
/// O(n log n) for M1. SIMD radix-select replaces this in M3.
pub fn topk_select(scores: &[f32], k: usize, scratch: &mut TopKScratch) {
    scratch.indexed.clear();
    scratch.indexed.reserve(scores.len());
    for (i, &v) in scores.iter().enumerate() {
        scratch.indexed.push((i as u32, v));
    }
    // Sort by score descending; ties broken by index ascending (stable tie-break).
    scratch.indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scratch.indexed.truncate(k);
    // Re-sort by index ascending for stable downstream output.
    scratch.indexed.sort_by_key(|&(i, _)| i);
}

use super::Backend;

pub static SCALAR: Backend = Backend { name: "scalar" };

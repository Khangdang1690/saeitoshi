//! Sparse decoder entry point. Scalar-only; SIMD decode would slot in
//! behind the same function once a gather-multiply-accumulate microkernel
//! is justified by profiling.

use crate::sae::{DecoderWeights, SparseOut};

/// Sparse decoder: `out[b, :] = b_dec + sum_{(f, v) in z[b]} v * W_dec[f, :]`.
///
/// Matches PyTorch's FP32 op order: row-major, f32 accumulators, no FMA
/// reassociation — keeps ≤1e-5 parity with SAELens.
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

//! Scalar reference impls used by the decode path and the legacy TopK
//! dispatcher. Both move out in the next commits — `decode_sparse` into
//! [`super::decoder`], `topk_select` removed entirely.

use crate::sae::{DecoderWeights, SparseOut};
use crate::sparsify::TopKScratch;

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
pub fn topk_select(scores: &[f32], k: usize, scratch: &mut TopKScratch) {
    scratch.indexed.clear();
    scratch.indexed.reserve(scores.len());
    for (i, &v) in scores.iter().enumerate() {
        scratch.indexed.push((i as u32, v));
    }
    scratch.indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scratch.indexed.truncate(k);
    scratch.indexed.sort_by_key(|&(i, _)| i);
}

//! Sparse decoder entry point. Currently scalar-only; SIMD decode is a
//! gather-multiply-accumulate that lives behind the same `Backend` field
//! when added in a follow-on.

use crate::sae::{DecoderWeights, SparseOut};

#[inline]
pub fn decode_sparse(z: &SparseOut, dec: &DecoderWeights, out: &mut [f32]) {
    super::scalar::decode_sparse(z, dec, out);
}

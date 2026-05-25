//! Sparse decoder entry point.

use crate::sae::{DecoderWeights, SparseOut};

#[inline]
pub fn decode_sparse(z: &SparseOut, dec: &DecoderWeights, out: &mut [f32]) {
    super::scalar::decode_sparse(z, dec, out);
}

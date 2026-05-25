//! TopK selection entry point. Scalar-only in M3 (the encoder matmul is the
//! dominant cost; SIMD radix-select is a future optimization).

use crate::sparsify::TopKScratch;

#[inline]
pub fn select_into(scores: &[f32], k: usize, scratch: &mut TopKScratch) {
    super::scalar::topk_select(scores, k, scratch);
}

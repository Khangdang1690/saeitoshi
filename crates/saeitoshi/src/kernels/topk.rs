//! TopK selection entry point.

use crate::sparsify::TopKScratch;

#[inline]
pub fn select_into(scores: &[f32], k: usize, scratch: &mut TopKScratch) {
    super::scalar::topk_select(scores, k, scratch);
}

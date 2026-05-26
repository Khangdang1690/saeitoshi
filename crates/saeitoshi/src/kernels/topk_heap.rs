//! Heap-based partial-sort TopK selection.
//!
//! Replaces the legacy full-sort path in `kernels::scalar::topk_select`
//! (O(d_sae log d_sae)) with a k-element min-heap partial sort
//! (O(d_sae log k)). At the v0.2 headline shape (d_sae=16384, k=32),
//! that's a ~3x algorithmic win on comparisons alone.
//!
//! Comparison is implemented via an integer-encoded ranking key
//! (`(value DESC, index ASC)` packed into a `u64`) so the inner loop
//! compiles to a single 64-bit compare with no closure overhead. LLVM
//! turns the predicate into `cmov`/`csel` on common targets, avoiding
//! mispredicts on the late-scan "no replace" branch.
//!
//! Output contract matches `scalar::topk_select` exactly:
//! - Select the k largest scores.
//! - Tie-break by index ascending.
//! - Write into `scratch.indexed` sorted by index ascending.
//!
//! See `crates/saeitoshi/tests/topk_parity.rs` for the parity gate.

use crate::sparsify::TopKScratch;

/// Map an `f32` to a `u32` whose unsigned ordering matches the float's
/// total order (NaN aside). Flips the sign bit for positives and all
/// bits for negatives.
#[inline(always)]
fn f32_to_ord(v: f32) -> u32 {
    let bits = v.to_bits();
    let sign_mask = ((bits as i32) >> 31) as u32;
    bits ^ (sign_mask | 0x8000_0000)
}

/// Pack `(value, index)` into a `u64` ranking key. Larger key == higher
/// rank under `(value DESC, then index ASC)`.
#[inline(always)]
fn rank_key(idx: u32, val: f32) -> u64 {
    ((f32_to_ord(val) as u64) << 32) | (!idx as u64)
}

#[inline(always)]
fn worse(a_idx: u32, a_val: f32, b_idx: u32, b_val: f32) -> bool {
    rank_key(a_idx, a_val) < rank_key(b_idx, b_val)
}

#[inline(always)]
fn better(a_idx: u32, a_val: f32, b_idx: u32, b_val: f32) -> bool {
    rank_key(a_idx, a_val) > rank_key(b_idx, b_val)
}

/// Restore the min-heap invariant by sifting `heap[i]` downward.
///
/// Heap layout: parent at `i`, children at `2i+1` and `2i+2`. Root
/// (`heap[0]`) holds the worst element of the current top-k — the
/// element a new candidate must beat to enter.
#[inline]
fn sift_down(heap: &mut [(u32, f32)], mut i: usize) {
    let n = heap.len();
    loop {
        let l = 2 * i + 1;
        if l >= n {
            return;
        }
        let r = l + 1;
        let (pi, pv) = heap[i];
        let (li, lv) = heap[l];

        let (mut worst_pos, mut wi, mut wv) = (l, li, lv);
        if r < n {
            let (ri, rv) = heap[r];
            if worse(ri, rv, li, lv) {
                worst_pos = r;
                wi = ri;
                wv = rv;
            }
        }

        if worse(wi, wv, pi, pv) {
            heap.swap(i, worst_pos);
            i = worst_pos;
        } else {
            return;
        }
    }
}

/// Bottom-up heapify: turn an arbitrary slice of `(idx, val)` pairs
/// into a min-heap (by `worse` ordering).
#[inline]
fn heapify(heap: &mut [(u32, f32)]) {
    let n = heap.len();
    if n <= 1 {
        return;
    }
    let mut i = n / 2;
    loop {
        i -= 1;
        sift_down(heap, i);
        if i == 0 {
            return;
        }
    }
}

/// Select the k largest entries from `scores`, write them sorted by
/// index ascending into `scratch.indexed`. Ties broken by index
/// ascending.
///
/// Equivalent in output to `kernels::scalar::topk_select`, but
/// algorithmically O(n log k) instead of O(n log n).
pub fn topk_select(scores: &[f32], k: usize, scratch: &mut TopKScratch) {
    let n = scores.len();
    let k = k.min(n);
    let buf = &mut scratch.indexed;
    buf.clear();

    if k == 0 {
        return;
    }

    buf.reserve(k);

    // Phase A: seed the heap with the first k entries in input order.
    for (i, &v) in scores.iter().take(k).enumerate() {
        buf.push((i as u32, v));
    }

    // Phase B: heapify, then scan the rest replacing the root when beaten.
    if k < n {
        heapify(buf);
        let heap: &mut [(u32, f32)] = buf;
        for (offset, &v) in scores[k..].iter().enumerate() {
            let i = (k + offset) as u32;
            let (root_idx, root_val) = heap[0];
            if better(i, v, root_idx, root_val) {
                heap[0] = (i, v);
                sift_down(heap, 0);
            }
        }
    }

    // Phase C: sort the top-k by index ascending (output contract).
    buf.sort_unstable_by_key(|&(idx, _)| idx);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut scratch = TopKScratch::new();
        topk_select(scores, k, &mut scratch);
        scratch.indexed
    }

    #[test]
    fn empty_k_returns_empty() {
        let out = run(&[1.0, 2.0, 3.0], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn k_equals_n_returns_all_in_index_order() {
        let out = run(&[3.0, 1.0, 4.0, 1.5, 9.0, 2.0], 6);
        let idx: Vec<u32> = out.iter().map(|&(i, _)| i).collect();
        assert_eq!(idx, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn k_greater_than_n_clamps_to_n() {
        let out = run(&[3.0, 1.0, 4.0], 10);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn matches_brief_example() {
        // top-3 in [1, 5, 2, 4, 3, 0.5, 7, 6] is {1: 5.0, 6: 7.0, 7: 6.0}.
        let out = run(&[1.0, 5.0, 2.0, 4.0, 3.0, 0.5, 7.0, 6.0], 3);
        let idx: Vec<u32> = out.iter().map(|&(i, _)| i).collect();
        assert_eq!(idx, vec![1, 6, 7]);
    }

    #[test]
    fn tie_break_lowest_indices_win() {
        // All equal -> pick the two lowest indices.
        let out = run(&[3.0, 3.0, 3.0, 3.0], 2);
        let idx: Vec<u32> = out.iter().map(|&(i, _)| i).collect();
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn matches_scalar_reference_on_random_input() {
        let mut scores: Vec<f32> = Vec::with_capacity(513);
        let mut state: u32 = 0xC0FFEE;
        for _ in 0..513 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((state >> 8) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
            scores.push(v);
        }
        for &k in &[1usize, 3, 32, 100, 512] {
            let mut scratch_a = TopKScratch::new();
            let mut scratch_b = TopKScratch::new();
            topk_select(&scores, k, &mut scratch_a);
            crate::kernels::scalar::topk_select(&scores, k, &mut scratch_b);
            assert_eq!(scratch_a.indexed, scratch_b.indexed, "k={k}");
        }
    }

    #[test]
    fn handles_negative_values_correctly() {
        let out = run(&[-1.0, -3.0, 2.0, -0.5, 5.0, -100.0, 0.0], 3);
        let idx: Vec<u32> = out.iter().map(|&(i, _)| i).collect();
        // Top-3: 5.0 (idx 4), 2.0 (idx 2), 0.0 (idx 6) -> index-asc [2, 4, 6].
        assert_eq!(idx, vec![2, 4, 6]);
    }
}

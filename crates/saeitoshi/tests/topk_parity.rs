//! TopK parity tests: heap-based partial sort must produce byte-identical
//! output to the legacy full-sort path across a grid of shapes and seeds.
//!
//! This is the load-bearing contract for the v0.2 redesign — the entire
//! Python parity test suite (vs `sae_lens`) and the existing Rust
//! `topk_tie_break_index_ascending` test depend on this equality.

#![cfg(feature = "topk-v2")]

use saeitoshi::sparsify::TopKScratch;

fn lcg_floats(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let raw = (state >> 40) as u32;
        let f = (raw as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        out.push(f * scale);
    }
    out
}

fn check_parity(d_sae: usize, k: usize, seed: u64) {
    let scores = lcg_floats(d_sae, seed, 1.0);
    let mut scratch_heap = TopKScratch::new();
    let mut scratch_ref = TopKScratch::new();
    saeitoshi::kernels::topk_heap::topk_select(&scores, k, &mut scratch_heap);
    saeitoshi::kernels::scalar::topk_select(&scores, k, &mut scratch_ref);
    assert_eq!(
        scratch_heap.indexed, scratch_ref.indexed,
        "TopK parity (d_sae={d_sae}, k={k}, seed={seed})"
    );
}

#[test]
fn small_shapes_grid() {
    let shapes = [(8usize, 1), (8, 3), (16, 4), (64, 4), (256, 16), (256, 100)];
    for (d_sae, k) in shapes {
        for seed in 1u64..6 {
            check_parity(d_sae, k, seed);
        }
    }
}

#[test]
fn headline_shape() {
    // The v0.2 headline: d_sae=16384, k=32.
    for seed in 1u64..6 {
        check_parity(16_384, 32, seed);
    }
}

#[test]
fn k_equals_d_sae() {
    for &d_sae in &[1usize, 8, 32, 256, 1024] {
        for seed in 1u64..4 {
            check_parity(d_sae, d_sae, seed);
        }
    }
}

#[test]
fn k_one() {
    for &d_sae in &[1usize, 8, 256, 16_384] {
        for seed in 1u64..4 {
            check_parity(d_sae, 1, seed);
        }
    }
}

#[test]
fn ties_everywhere() {
    // Every score equal — pure tie-break, only ordering by index matters.
    let scores = vec![3.5_f32; 1024];
    let mut heap = TopKScratch::new();
    let mut refs = TopKScratch::new();
    for &k in &[1usize, 8, 32, 100, 1024] {
        saeitoshi::kernels::topk_heap::topk_select(&scores, k, &mut heap);
        saeitoshi::kernels::scalar::topk_select(&scores, k, &mut refs);
        assert_eq!(heap.indexed, refs.indexed, "ties k={k}");
        // Sanity: must be exactly k lowest indices.
        let want: Vec<u32> = (0..k as u32).collect();
        let got: Vec<u32> = heap.indexed.iter().map(|&(i, _)| i).collect();
        assert_eq!(got, want, "ties k={k} indices");
    }
}

#[test]
fn partial_ties_at_boundary() {
    // Several scores tied at the cut: tie-break by index must keep the
    // lowest indices of the tied group.
    // Scores: [5, 5, 5, 1, 1, 5, 5, 5] (eight values, four 5s in
    // various positions, k=3 should pick indices [0, 1, 2] — the first
    // three 5s in index order).
    let scores: Vec<f32> = vec![5.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0];
    let mut heap = TopKScratch::new();
    let mut refs = TopKScratch::new();
    saeitoshi::kernels::topk_heap::topk_select(&scores, 3, &mut heap);
    saeitoshi::kernels::scalar::topk_select(&scores, 3, &mut refs);
    assert_eq!(heap.indexed, refs.indexed);
    let got: Vec<u32> = heap.indexed.iter().map(|&(i, _)| i).collect();
    assert_eq!(got, vec![0, 1, 2]);
}

#[test]
fn dispatch_resolves_to_heap_by_default() {
    // `SAEITOSHI_TOPK` unset (or anything but "legacy") → heap path.
    assert_eq!(saeitoshi::kernels::topk::backend_name(), "heap");
}

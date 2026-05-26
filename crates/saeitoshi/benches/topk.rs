//! TopK selection benchmark — measures `topk_select` in isolation,
//! independent of the matmul. v0.2 primary signal.
//!
//! Compares the legacy full-sort against the heap-based partial sort
//! across a range of `(d_sae, k)` pairs. Batch dimension is simulated
//! by running the same call back-to-back inside criterion's iteration
//! loop, so the per-row cost is what the measurement reflects.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use saeitoshi::sparsify::TopKScratch;

fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let raw = (s >> 40) as u32;
            ((raw as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

type TopkFn = fn(&[f32], usize, &mut TopKScratch);

fn bench_one(c: &mut Criterion, name: &str, f: TopkFn, d_sae: usize, k: usize) {
    let scores = lcg(d_sae, 0xC0FFEE, 1.0);
    let mut scratch = TopKScratch::new();
    c.bench_with_input(
        BenchmarkId::new(name, format!("d_sae={d_sae} k={k}")),
        &(d_sae, k),
        |b, _| {
            b.iter(|| {
                f(black_box(&scores), black_box(k), black_box(&mut scratch));
            })
        },
    );
}

fn bench_topk(c: &mut Criterion) {
    let shapes = [(8192usize, 32), (16_384, 32), (16_384, 64), (16_384, 1024)];

    for (d_sae, k) in shapes {
        bench_one(
            c,
            "legacy_full_sort",
            saeitoshi::kernels::scalar::topk_select,
            d_sae,
            k,
        );
        #[cfg(feature = "topk-v2")]
        bench_one(
            c,
            "heap_partial_sort",
            saeitoshi::kernels::topk_heap::topk_select,
            d_sae,
            k,
        );
    }
}

criterion_group!(benches, bench_topk);
criterion_main!(benches);

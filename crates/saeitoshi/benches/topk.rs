//! TopK selection benchmark — measures `topk_select` in isolation,
//! independent of the matmul.

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

fn bench_topk(c: &mut Criterion) {
    let shapes = [(8192usize, 32), (16_384, 32), (16_384, 64), (16_384, 1024)];

    for (d_sae, k) in shapes {
        let scores = lcg(d_sae, 0xC0FFEE, 1.0);
        let mut scratch = TopKScratch::new();
        c.bench_with_input(
            BenchmarkId::new("heap_partial_sort", format!("d_sae={d_sae} k={k}")),
            &(d_sae, k),
            |b, _| {
                b.iter(|| {
                    saeitoshi::kernels::topk::topk_select(
                        black_box(&scores),
                        black_box(k),
                        black_box(&mut scratch),
                    );
                })
            },
        );
    }
}

criterion_group!(benches, bench_topk);
criterion_main!(benches);

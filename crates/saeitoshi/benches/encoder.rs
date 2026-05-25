//! Encoder benchmarks. Populated in M3 once SIMD kernels land — each backend
//! benchmarked on the same problem so we can compare AVX2 / AVX-512 / NEON /
//! scalar head-to-head per SAE size.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, placeholder);
criterion_main!(benches);

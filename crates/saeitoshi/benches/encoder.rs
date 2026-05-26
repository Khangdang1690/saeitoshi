//! Encoder backend benchmark. Compares scalar vs each SIMD backend that
//! the current CPU supports, on a mid-size SAE (d_in ≈ 1024, d_sae ≈ 16K).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use saeitoshi::backends::SCALAR;
use saeitoshi::kernels::{select_backend, Backend};
use saeitoshi::sae::EncoderWeights;

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

fn run_backend(
    c: &mut Criterion,
    backend: &'static Backend,
    d_in: usize,
    d_sae: usize,
    batch: usize,
) {
    let enc = EncoderWeights {
        w_enc: lcg(d_sae * d_in, 1, 0.1).into_boxed_slice(),
        b_enc: lcg(d_sae, 2, 0.01).into_boxed_slice(),
        d_in,
        d_sae,
    };
    let x = lcg(batch * d_in, 3, 1.0);
    let mut out = vec![0.0f32; batch * d_sae];

    c.bench_with_input(
        BenchmarkId::new(backend.name, format!("d_in={d_in} d_sae={d_sae} B={batch}")),
        &(d_in, d_sae, batch),
        |b, _| {
            b.iter(|| {
                (backend.encode_f32)(black_box(&x), black_box(&enc), black_box(&mut out), batch);
            })
        },
    );
}

fn bench_encoder(c: &mut Criterion) {
    let d_in = 1024usize;
    let d_sae = 16_384usize;
    let batch = 8usize;

    run_backend(c, &SCALAR, d_in, d_sae, batch);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            run_backend(c, &saeitoshi::backends::AVX2, d_in, d_sae, batch);
        }
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            run_backend(c, &saeitoshi::backends::AVX512, d_in, d_sae, batch);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            run_backend(c, &saeitoshi::backends::NEON, d_in, d_sae, batch);
        }
    }

    eprintln!(
        "auto-selected backend on this CPU: {}",
        select_backend().name
    );
}

criterion_group!(benches, bench_encoder);
criterion_main!(benches);

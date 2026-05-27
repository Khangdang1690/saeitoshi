//! Encoder backend benchmark. Runs the tiled-scalar reference plus each
//! tiled-SIMD backend the current CPU supports, on a mid-size SAE
//! (d_in ≈ 1024, d_sae ≈ 16K) and a headline shape (d_in=2048,
//! d_sae=16384, batch=512). Skip the headline shape with
//! `SAEITOSHI_SKIP_HEADLINE_BENCH=1` while iterating.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use saeitoshi::backends::SCALAR_TILED;
use saeitoshi::kernels::{select_backend, Backend};
use saeitoshi::sae::{EncoderWeights, WeightLayout};

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

fn run_tiled(
    c: &mut Criterion,
    backend: &'static Backend,
    d_in: usize,
    d_sae: usize,
    batch: usize,
) {
    use saeitoshi::kernels::gemm::{pack::repack, DEFAULT_M_R};
    let mut enc = EncoderWeights {
        w_enc: lcg(d_sae * d_in, 1, 0.1).into_boxed_slice(),
        b_enc: lcg(d_sae, 2, 0.01).into_boxed_slice(),
        d_in,
        d_sae,
        layout: WeightLayout::RowMajor,
    };
    repack(&mut enc, DEFAULT_M_R);
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

    run_tiled(c, &SCALAR_TILED, d_in, d_sae, batch);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            run_tiled(c, &saeitoshi::backends::AVX2_TILED, d_in, d_sae, batch);
        }
        if std::is_x86_feature_detected!("avx512f") {
            run_tiled(c, &saeitoshi::backends::AVX512_TILED, d_in, d_sae, batch);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            run_tiled(c, &saeitoshi::backends::NEON_TILED, d_in, d_sae, batch);
        }
    }

    // Headline shape: bigger d_in + batch so M-block rayon partitioning
    // and per-thread x-panel sharing dominate the measurement.
    #[cfg(target_arch = "x86_64")]
    if std::env::var("SAEITOSHI_SKIP_HEADLINE_BENCH").is_err() {
        let h_d_in = 2048usize;
        let h_d_sae = 16_384usize;
        let h_batch = 512usize;
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            run_tiled(
                c,
                &saeitoshi::backends::AVX2_TILED,
                h_d_in,
                h_d_sae,
                h_batch,
            );
        }
        if std::is_x86_feature_detected!("avx512f") {
            run_tiled(
                c,
                &saeitoshi::backends::AVX512_TILED,
                h_d_in,
                h_d_sae,
                h_batch,
            );
        }
    }

    eprintln!(
        "auto-selected backend on this CPU: {}",
        select_backend().name
    );
}

criterion_group!(benches, bench_encoder);
criterion_main!(benches);

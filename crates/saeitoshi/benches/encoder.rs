//! Encoder backend benchmark. Compares scalar vs each SIMD backend that
//! the current CPU supports, on a mid-size SAE (d_in ≈ 1024, d_sae ≈ 16K).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use saeitoshi::backends::SCALAR;
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
        layout: WeightLayout::RowMajor,
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

    #[cfg(feature = "perf-v2")]
    run_tiled_scalar(c, d_in, d_sae, batch);

    // Single-thread headline shape for the tiled SIMD backends. M4 adds
    // the M-block rayon split; until then, this bench shows the register-
    // blocking win in isolation. Skip with `SAEITOSHI_SKIP_HEADLINE_BENCH=1`
    // when iterating on the kernel under criterion's defaults.
    #[cfg(all(feature = "perf-v2", target_arch = "x86_64"))]
    if std::env::var("SAEITOSHI_SKIP_HEADLINE_BENCH").is_err() {
        let h_d_in = 2048usize;
        let h_d_sae = 16_384usize;
        let h_batch = 512usize;
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            run_tiled_simd(
                c,
                &saeitoshi::backends::AVX2_TILED,
                h_d_in,
                h_d_sae,
                h_batch,
            );
        }
        if std::is_x86_feature_detected!("avx512f") {
            run_tiled_simd(
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

#[cfg(all(feature = "perf-v2", target_arch = "x86_64"))]
fn run_tiled_simd(
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

#[cfg(feature = "perf-v2")]
fn run_tiled_scalar(c: &mut Criterion, d_in: usize, d_sae: usize, batch: usize) {
    use saeitoshi::backends::SCALAR_TILED;
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
        BenchmarkId::new(
            SCALAR_TILED.name,
            format!("d_in={d_in} d_sae={d_sae} B={batch}"),
        ),
        &(d_in, d_sae, batch),
        |b, _| {
            b.iter(|| {
                (SCALAR_TILED.encode_f32)(
                    black_box(&x),
                    black_box(&enc),
                    black_box(&mut out),
                    batch,
                );
            })
        },
    );
}

criterion_group!(benches, bench_encoder);
criterion_main!(benches);

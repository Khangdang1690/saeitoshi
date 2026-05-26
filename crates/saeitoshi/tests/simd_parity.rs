//! SIMD-vs-scalar parity tests.
//!
//! For every available backend on the current CPU, run the encoder on
//! randomly-generated weights and inputs across a range of `d_in` /
//! `d_sae` / `batch` sizes. SIMD must agree with scalar to within 1e-5
//! absolute error — this is the launch-blocking parity gate.

use saeitoshi::backends::SCALAR;
use saeitoshi::kernels::{select_backend, Backend};
use saeitoshi::sae::{EncoderWeights, WeightLayout};

fn lcg_floats(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map upper 24 bits to ±1, then scale.
        let raw = (state >> 40) as u32;
        let f = (raw as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        out.push(f * scale);
    }
    out
}

fn make_encoder(d_in: usize, d_sae: usize, seed: u64) -> EncoderWeights {
    EncoderWeights {
        w_enc: lcg_floats(d_sae * d_in, seed, 0.1).into_boxed_slice(),
        b_enc: lcg_floats(d_sae, seed.wrapping_add(1), 0.01).into_boxed_slice(),
        d_in,
        d_sae,
        layout: WeightLayout::RowMajor,
    }
}

fn run_backend(backend: &Backend, x: &[f32], enc: &EncoderWeights, batch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * enc.d_sae];
    (backend.encode_f32)(x, enc, &mut out, batch);
    out
}

fn check_parity(name: &str, backend: &Backend, d_in: usize, d_sae: usize, batch: usize, seed: u64) {
    let enc = make_encoder(d_in, d_sae, seed);
    let x = lcg_floats(batch * d_in, seed.wrapping_add(100), 1.0);
    let baseline = run_backend(&SCALAR, &x, &enc, batch);
    let candidate = run_backend(backend, &x, &enc, batch);
    let max_err = baseline
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-5,
        "{name} parity (d_in={d_in}, d_sae={d_sae}, batch={batch}, seed={seed}): max abs error {max_err}",
    );
}

fn check_backend_across_sizes(name: &str, backend: &Backend) {
    // A spread of d_in values: ones that fit cleanly into 4 / 8 / 16 lane
    // SIMD registers, plus odd sizes that exercise the scalar tail.
    let sizes = [
        (4usize, 8),
        (7, 16),
        (8, 16),
        (15, 32),
        (16, 32),
        (33, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (1024, 2048),
    ];
    for (i, &(d_in, d_sae)) in sizes.iter().enumerate() {
        for &batch in &[1usize, 4, 8] {
            check_parity(name, backend, d_in, d_sae, batch, 42 + i as u64);
        }
    }
}

#[test]
fn scalar_self_parity() {
    // Sanity: scalar against itself is bit-exact.
    check_backend_across_sizes("scalar", &SCALAR);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_parity_with_scalar() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        eprintln!("skipping: AVX2/FMA not available on this CPU");
        return;
    }
    check_backend_across_sizes("avx2", &saeitoshi::backends::AVX2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_parity_with_scalar() {
    if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512bw") {
        eprintln!("skipping: AVX-512 not available on this CPU");
        return;
    }
    check_backend_across_sizes("avx512", &saeitoshi::backends::AVX512);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_parity_with_scalar() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        eprintln!("skipping: NEON not available on this CPU");
        return;
    }
    check_backend_across_sizes("neon", &saeitoshi::backends::NEON);
}

#[test]
fn auto_selected_backend_matches_scalar() {
    let backend = select_backend();
    eprintln!("auto-selected backend: {}", backend.name);
    check_backend_across_sizes(backend.name, backend);
}

// ----- perf-v2 (tiled GEMM) parity -----
//
// The scalar tiled backend reads W via the packed-panel layout but does the
// same f-outer / b-middle / k-inner reduction as the row-major scalar, so
// per-output FP arithmetic is bit-exact. We still check at 1e-5 (the
// launch-blocking gate) — bit-exact is a stricter property we get for free
// at this commit and lose in M3 when the AVX2 microkernel reorders K.

#[cfg(feature = "perf-v2")]
fn make_packed_encoder(d_in: usize, d_sae: usize, seed: u64, m_r: usize) -> EncoderWeights {
    let mut enc = make_encoder(d_in, d_sae, seed);
    saeitoshi::kernels::gemm::pack::repack(&mut enc, m_r);
    enc
}

#[cfg(feature = "perf-v2")]
fn check_tiled_parity(name: &str, d_in: usize, d_sae: usize, batch: usize, seed: u64) {
    use saeitoshi::backends::SCALAR_TILED;
    use saeitoshi::kernels::gemm::DEFAULT_M_R;
    let row_enc = make_encoder(d_in, d_sae, seed);
    let packed_enc = make_packed_encoder(d_in, d_sae, seed, DEFAULT_M_R);
    let x = lcg_floats(batch * d_in, seed.wrapping_add(100), 1.0);
    let baseline = run_backend(&SCALAR, &x, &row_enc, batch);
    let candidate = run_backend(&SCALAR_TILED, &x, &packed_enc, batch);
    let max_err = baseline
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-5,
        "{name} parity (d_in={d_in}, d_sae={d_sae}, batch={batch}, seed={seed}, m_r={}): \
         max abs error {max_err}",
        DEFAULT_M_R,
    );
}

#[cfg(feature = "perf-v2")]
#[test]
fn scalar_tiled_parity_with_scalar_standard_sizes() {
    // Same shape grid as `check_backend_across_sizes` so we cover odd d_in
    // (scalar tail), clean SIMD-aligned d_in, and d_sae values that span
    // partial trailing panels (e.g., 32 + 33 features at m_r=16).
    let sizes = [
        (4usize, 8),
        (7, 16),
        (8, 16),
        (15, 32),
        (16, 32),
        (33, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (1024, 2048),
    ];
    for (i, &(d_in, d_sae)) in sizes.iter().enumerate() {
        for &batch in &[1usize, 4, 8] {
            check_tiled_parity("scalar_tiled", d_in, d_sae, batch, 42 + i as u64);
        }
    }
}

#[cfg(all(feature = "perf-v2", target_arch = "x86_64"))]
fn check_tiled_simd_parity(
    name: &str,
    backend: &Backend,
    d_in: usize,
    d_sae: usize,
    batch: usize,
    seed: u64,
) {
    use saeitoshi::kernels::gemm::DEFAULT_M_R;
    let row_enc = make_encoder(d_in, d_sae, seed);
    let packed_enc = make_packed_encoder(d_in, d_sae, seed, DEFAULT_M_R);
    let x = lcg_floats(batch * d_in, seed.wrapping_add(100), 1.0);
    let baseline = run_backend(&SCALAR, &x, &row_enc, batch);
    let candidate = run_backend(backend, &x, &packed_enc, batch);
    let max_err = baseline
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-5,
        "{name} parity (d_in={d_in}, d_sae={d_sae}, batch={batch}, seed={seed}): \
         max abs error {max_err}",
    );
}

#[cfg(all(feature = "perf-v2", target_arch = "x86_64"))]
fn check_tiled_simd_across_sizes(name: &str, backend: &Backend) {
    // Standard grid plus a wider d_in to stress the K loop, and batch
    // values that exercise both full N_R tiles and partial-batch tails.
    let sizes = [
        (4usize, 16),
        (7, 16),
        (8, 32),
        (16, 32),
        (33, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (1024, 2048),
        (513, 96),  // odd d_in + d_sae with partial trailing panel
    ];
    for (i, &(d_in, d_sae)) in sizes.iter().enumerate() {
        // Batch 1 / 5 / 6 / 12 / 13 covers below, partial, exact, exact, and
        // partial fills of both N_R=6 (AVX2) and N_R=12 (AVX-512) tiles.
        for &batch in &[1usize, 5, 6, 12, 13] {
            check_tiled_simd_parity(name, backend, d_in, d_sae, batch, 300 + i as u64);
        }
    }
}

#[cfg(all(feature = "perf-v2", target_arch = "x86_64"))]
#[test]
fn avx2_tiled_parity_with_scalar() {
    use saeitoshi::backends::AVX2_TILED;
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        eprintln!("skipping: AVX2/FMA not available on this CPU");
        return;
    }
    check_tiled_simd_across_sizes("avx2_tiled", &AVX2_TILED);
}

#[cfg(all(feature = "perf-v2", target_arch = "x86_64"))]
#[test]
fn avx512_tiled_parity_with_scalar() {
    use saeitoshi::backends::AVX512_TILED;
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: AVX-512F not available on this CPU");
        return;
    }
    check_tiled_simd_across_sizes("avx512_tiled", &AVX512_TILED);
}

#[cfg(feature = "perf-v2")]
#[test]
fn scalar_tiled_parity_awkward_d_in() {
    // Stress tail handling and packing-padded panels — these are the
    // shapes most likely to surface microkernel tail bugs in M3.
    let awkward_d_ins = [1usize, 7, 511, 513, 1023, 1024, 1025, 2049];
    for (i, &d_in) in awkward_d_ins.iter().enumerate() {
        // Pick d_sae values that exercise partial trailing panels at m_r=16:
        //   17 → panel0 full, panel1 = 1 used + 15 padded
        //   31 → panel0 full, panel1 = 15 used + 1 padded
        //   32 → panel0 full, panel1 full (no padding)
        for &d_sae in &[17usize, 31, 32, 64] {
            check_tiled_parity("scalar_tiled", d_in, d_sae, 2, 200 + i as u64);
        }
    }
}

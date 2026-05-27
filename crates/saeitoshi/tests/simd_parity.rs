//! Tiled-GEMM parity tests.
//!
//! For every tiled backend available on the current CPU, run the encoder
//! on randomly-generated weights and inputs across a range of `d_in` /
//! `d_sae` / `batch` sizes. Each backend must agree with the scalar-tiled
//! reference (same packed layout, scalar FMAs) to within 1e-5 absolute
//! error — the launch-blocking parity gate.

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

fn make_packed_encoder(d_in: usize, d_sae: usize, seed: u64, m_r: usize) -> EncoderWeights {
    let mut enc = make_encoder(d_in, d_sae, seed);
    saeitoshi::kernels::gemm::pack::repack(&mut enc, m_r);
    enc
}

fn run_backend(backend: &Backend, x: &[f32], enc: &EncoderWeights, batch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * enc.d_sae];
    (backend.encode_f32)(x, enc, &mut out, batch);
    out
}

fn check_tiled_parity(
    name: &str,
    backend: &Backend,
    d_in: usize,
    d_sae: usize,
    batch: usize,
    seed: u64,
) {
    use saeitoshi::kernels::gemm::DEFAULT_M_R;
    let packed_enc = make_packed_encoder(d_in, d_sae, seed, DEFAULT_M_R);
    let x = lcg_floats(batch * d_in, seed.wrapping_add(100), 1.0);
    let baseline = run_backend(&SCALAR, &x, &packed_enc, batch);
    let candidate = run_backend(backend, &x, &packed_enc, batch);
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
        (513, 96), // odd d_in + d_sae with partial trailing panel
    ];
    for (i, &(d_in, d_sae)) in sizes.iter().enumerate() {
        // Batch 1 / 5 / 6 / 12 / 13 covers below, partial, exact, exact, and
        // partial fills of both N_R=6 (AVX2) and N_R=12 (AVX-512) tiles.
        for &batch in &[1usize, 5, 6, 12, 13] {
            check_tiled_parity(name, backend, d_in, d_sae, batch, 300 + i as u64);
        }
    }
}

#[test]
fn auto_selected_backend_matches_scalar_tiled() {
    let backend = select_backend();
    eprintln!("auto-selected backend: {}", backend.name);
    check_tiled_simd_across_sizes(backend.name, backend);
}

#[test]
fn scalar_tiled_self_parity_standard_sizes() {
    // Sanity: scalar-tiled against itself is bit-exact. Same shape grid
    // we use for the SIMD variants, including odd d_in (scalar tail) and
    // d_sae values that span partial trailing panels.
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
        (513, 96),
    ];
    for (i, &(d_in, d_sae)) in sizes.iter().enumerate() {
        for &batch in &[1usize, 5, 6, 12, 13] {
            check_tiled_parity("scalar_tiled", &SCALAR, d_in, d_sae, batch, 200 + i as u64);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_tiled_parity_with_scalar() {
    use saeitoshi::backends::AVX2;
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        eprintln!("skipping: AVX2/FMA not available on this CPU");
        return;
    }
    check_tiled_simd_across_sizes("avx2_tiled", &AVX2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_tiled_parity_with_scalar() {
    use saeitoshi::backends::AVX512;
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: AVX-512F not available on this CPU");
        return;
    }
    check_tiled_simd_across_sizes("avx512_tiled", &AVX512);
}

/// Tiled backends do their own M-block rayon split internally. Different
/// thread counts → different task partitioning → different timing of
/// writes to disjoint columns of `pre_acts`. We need parity to hold at
/// every thread count we ship (1/2/4/24) to surface false-sharing bugs,
/// alignment bugs in the partial-panel scalar fallback, and any subtle
/// races we might have missed in the `AtomicPtr` handoff.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_tiled_parity_across_thread_counts() {
    use rayon::ThreadPoolBuilder;
    use saeitoshi::backends::AVX2;

    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        eprintln!("skipping: AVX2/FMA not available on this CPU");
        return;
    }

    // Shape large enough to span multiple M-blocks (M_C = 512, m_r = 16 →
    // 32 panels per M-block, so d_sae = 2048 → 4 M-blocks; 24 threads
    // exercises work-stealing on the trailing tasks).
    let d_in = 256usize;
    let d_sae = 2048usize;
    let batch = 12usize;

    for &num_threads in &[1usize, 2, 4, 24] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("rayon pool build");

        pool.install(|| {
            check_tiled_parity(
                &format!("avx2_tiled[threads={num_threads}]"),
                &AVX2,
                d_in,
                d_sae,
                batch,
                7777 + num_threads as u64,
            );
        });
    }
}

#[test]
fn scalar_tiled_parity_awkward_d_in() {
    // Stress tail handling and packing-padded panels.
    let awkward_d_ins = [1usize, 7, 511, 513, 1023, 1024, 1025, 2049];
    for (i, &d_in) in awkward_d_ins.iter().enumerate() {
        // Pick d_sae values that exercise partial trailing panels at m_r=16:
        //   17 → panel0 full, panel1 = 1 used + 15 padded
        //   31 → panel0 full, panel1 = 15 used + 1 padded
        //   32 → panel0 full, panel1 full (no padding)
        for &d_sae in &[17usize, 31, 32, 64] {
            check_tiled_parity("scalar_tiled", &SCALAR, d_in, d_sae, 2, 200 + i as u64);
        }
    }
}

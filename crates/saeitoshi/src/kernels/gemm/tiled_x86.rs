//! AVX2 (16x6) and AVX-512 (16x12) tiled GEMM encoders.
//!
//! Both microkernels share the same packed-W layout (M_R = 16), so a single
//! pack at SAE construction time (M4) feeds either backend without
//! repacking. The plan originally specified AVX-512 at 32x12; we land 16x12
//! here so packing infra is shared, and revisit the 32x12 variant as a
//! follow-up once the headline benchmark numbers are in.
//!
//! Numerical contract: ≤1e-5 max-abs-error vs the row-major scalar
//! reference (`kernels::scalar`). The microkernels reorder K reduction by
//! lane (each ymm/zmm lane carries its own FMA chain), but per-lane the
//! reduction is left-to-right in K — same as scalar. Cross-lane reordering
//! happens only inside the FMA itself (mul+add fused into one rounding,
//! vs scalar's two roundings). That's the entire FP delta; the design
//! analysis (docs/perf-analysis.md) bounds it well within 1e-5 for our
//! d_in range.

#![allow(unsafe_code)]

use std::arch::x86_64::*;

use crate::kernels::Backend;
use crate::sae::{EncoderWeights, WeightLayout};

use super::pack::{packed_len, panel_count};

// ---------------- shared register-tile constants ----------------

/// AVX2 register tile: 16 features wide (2 ymm), 6 batch rows.
///
/// Inner loop accounting per K step: 6 broadcasts of `x[b, k]` + 2 ymm loads
/// of packed W + 12 FMAs into the 12-ymm accumulator tile. 14 load uops
/// against 12 FMAs — comfortably FMA-bound on Raptor Lake (3 load ports,
/// 2 FMA ports). 12 acc + 2 W + 1 broadcast = 15 live ymm; fits in 16 with
/// one register to spare for the next-K weight prefetch (M5).
const M_R_AVX2: usize = 16;
const N_R_AVX2: usize = 6;

/// AVX-512 register tile: 16 features wide (1 zmm), 12 batch rows.
///
/// Inner loop: 12 broadcasts + 1 zmm load + 12 FMAs per K. 12 acc + 1 W +
/// 1 broadcast = 14 live zmm; fits in 32 with 18 spare. We do not use the
/// theoretical max N_R because batch is often smaller than 24 in practice;
/// 12 hits the FMA peak (12 FMAs / 2 ports = 6 cycles) without blowing
/// out the tail batch handling.
const M_R_AVX512: usize = 16;
const N_R_AVX512: usize = 12;

// ---------------- AVX2 microkernel ----------------

/// Compute one full `M_R_AVX2 × N_R_AVX2` (= 16 × 6) tile of pre_acts.
///
/// All pointers must be valid for the implied accesses; the caller is
/// responsible for ensuring `f0 + 16 ≤ d_sae`, `b0 + 6 ≤ batch`, and
/// `k_count == d_in`. AVX2+FMA must be available — gated by the
/// `#[target_feature]` attribute, asserted at backend selection.
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_16x6(
    packed_w_panel: *const f32,
    x_base: *const f32,
    d_in: usize,
    pre_acts_tile: *mut f32,
    out_row_stride: usize,
    bias_base: *const f32,
) {
    let bias_lo = _mm256_loadu_ps(bias_base);
    let bias_hi = _mm256_loadu_ps(bias_base.add(8));

    // Twelve named accumulators. LLVM SSA puts each in its own ymm; bias_lo /
    // bias_hi are dead after init, so their registers are reclaimed.
    let mut a00 = bias_lo;
    let mut a01 = bias_hi;
    let mut a10 = bias_lo;
    let mut a11 = bias_hi;
    let mut a20 = bias_lo;
    let mut a21 = bias_hi;
    let mut a30 = bias_lo;
    let mut a31 = bias_hi;
    let mut a40 = bias_lo;
    let mut a41 = bias_hi;
    let mut a50 = bias_lo;
    let mut a51 = bias_hi;

    for k in 0..d_in {
        // Load the 16 packed W floats for this k as 2 ymm vectors.
        let w_lo = _mm256_loadu_ps(packed_w_panel.add(k * M_R_AVX2));
        let w_hi = _mm256_loadu_ps(packed_w_panel.add(k * M_R_AVX2 + 8));

        // Each broadcast lives only across its 2 FMAs — single ymm reused.
        let x0 = _mm256_set1_ps(*x_base.add(k));
        a00 = _mm256_fmadd_ps(x0, w_lo, a00);
        a01 = _mm256_fmadd_ps(x0, w_hi, a01);

        let x1 = _mm256_set1_ps(*x_base.add(d_in + k));
        a10 = _mm256_fmadd_ps(x1, w_lo, a10);
        a11 = _mm256_fmadd_ps(x1, w_hi, a11);

        let x2 = _mm256_set1_ps(*x_base.add(2 * d_in + k));
        a20 = _mm256_fmadd_ps(x2, w_lo, a20);
        a21 = _mm256_fmadd_ps(x2, w_hi, a21);

        let x3 = _mm256_set1_ps(*x_base.add(3 * d_in + k));
        a30 = _mm256_fmadd_ps(x3, w_lo, a30);
        a31 = _mm256_fmadd_ps(x3, w_hi, a31);

        let x4 = _mm256_set1_ps(*x_base.add(4 * d_in + k));
        a40 = _mm256_fmadd_ps(x4, w_lo, a40);
        a41 = _mm256_fmadd_ps(x4, w_hi, a41);

        let x5 = _mm256_set1_ps(*x_base.add(5 * d_in + k));
        a50 = _mm256_fmadd_ps(x5, w_lo, a50);
        a51 = _mm256_fmadd_ps(x5, w_hi, a51);
    }

    _mm256_storeu_ps(pre_acts_tile, a00);
    _mm256_storeu_ps(pre_acts_tile.add(8), a01);
    _mm256_storeu_ps(pre_acts_tile.add(out_row_stride), a10);
    _mm256_storeu_ps(pre_acts_tile.add(out_row_stride + 8), a11);
    _mm256_storeu_ps(pre_acts_tile.add(2 * out_row_stride), a20);
    _mm256_storeu_ps(pre_acts_tile.add(2 * out_row_stride + 8), a21);
    _mm256_storeu_ps(pre_acts_tile.add(3 * out_row_stride), a30);
    _mm256_storeu_ps(pre_acts_tile.add(3 * out_row_stride + 8), a31);
    _mm256_storeu_ps(pre_acts_tile.add(4 * out_row_stride), a40);
    _mm256_storeu_ps(pre_acts_tile.add(4 * out_row_stride + 8), a41);
    _mm256_storeu_ps(pre_acts_tile.add(5 * out_row_stride), a50);
    _mm256_storeu_ps(pre_acts_tile.add(5 * out_row_stride + 8), a51);
}

// ---------------- AVX-512 microkernel ----------------

#[target_feature(enable = "avx512f")]
unsafe fn microkernel_16x12_avx512(
    packed_w_panel: *const f32,
    x_base: *const f32,
    d_in: usize,
    pre_acts_tile: *mut f32,
    out_row_stride: usize,
    bias_base: *const f32,
) {
    let bias = _mm512_loadu_ps(bias_base);
    let mut a0 = bias;
    let mut a1 = bias;
    let mut a2 = bias;
    let mut a3 = bias;
    let mut a4 = bias;
    let mut a5 = bias;
    let mut a6 = bias;
    let mut a7 = bias;
    let mut a8 = bias;
    let mut a9 = bias;
    let mut a10 = bias;
    let mut a11 = bias;

    for k in 0..d_in {
        let w = _mm512_loadu_ps(packed_w_panel.add(k * M_R_AVX512));

        let x0 = _mm512_set1_ps(*x_base.add(k));
        a0 = _mm512_fmadd_ps(x0, w, a0);
        let x1 = _mm512_set1_ps(*x_base.add(d_in + k));
        a1 = _mm512_fmadd_ps(x1, w, a1);
        let x2 = _mm512_set1_ps(*x_base.add(2 * d_in + k));
        a2 = _mm512_fmadd_ps(x2, w, a2);
        let x3 = _mm512_set1_ps(*x_base.add(3 * d_in + k));
        a3 = _mm512_fmadd_ps(x3, w, a3);
        let x4 = _mm512_set1_ps(*x_base.add(4 * d_in + k));
        a4 = _mm512_fmadd_ps(x4, w, a4);
        let x5 = _mm512_set1_ps(*x_base.add(5 * d_in + k));
        a5 = _mm512_fmadd_ps(x5, w, a5);
        let x6 = _mm512_set1_ps(*x_base.add(6 * d_in + k));
        a6 = _mm512_fmadd_ps(x6, w, a6);
        let x7 = _mm512_set1_ps(*x_base.add(7 * d_in + k));
        a7 = _mm512_fmadd_ps(x7, w, a7);
        let x8 = _mm512_set1_ps(*x_base.add(8 * d_in + k));
        a8 = _mm512_fmadd_ps(x8, w, a8);
        let x9 = _mm512_set1_ps(*x_base.add(9 * d_in + k));
        a9 = _mm512_fmadd_ps(x9, w, a9);
        let x10 = _mm512_set1_ps(*x_base.add(10 * d_in + k));
        a10 = _mm512_fmadd_ps(x10, w, a10);
        let x11 = _mm512_set1_ps(*x_base.add(11 * d_in + k));
        a11 = _mm512_fmadd_ps(x11, w, a11);
    }

    _mm512_storeu_ps(pre_acts_tile, a0);
    _mm512_storeu_ps(pre_acts_tile.add(out_row_stride), a1);
    _mm512_storeu_ps(pre_acts_tile.add(2 * out_row_stride), a2);
    _mm512_storeu_ps(pre_acts_tile.add(3 * out_row_stride), a3);
    _mm512_storeu_ps(pre_acts_tile.add(4 * out_row_stride), a4);
    _mm512_storeu_ps(pre_acts_tile.add(5 * out_row_stride), a5);
    _mm512_storeu_ps(pre_acts_tile.add(6 * out_row_stride), a6);
    _mm512_storeu_ps(pre_acts_tile.add(7 * out_row_stride), a7);
    _mm512_storeu_ps(pre_acts_tile.add(8 * out_row_stride), a8);
    _mm512_storeu_ps(pre_acts_tile.add(9 * out_row_stride), a9);
    _mm512_storeu_ps(pre_acts_tile.add(10 * out_row_stride), a10);
    _mm512_storeu_ps(pre_acts_tile.add(11 * out_row_stride), a11);
}

// ---------------- scalar fallback for tail tiles ----------------

/// Sharing `*mut f32` across rayon tasks for the M-block parallel pass.
///
/// The parallel iteration in [`encode_f32_avx2_tiled`] / `_avx512_tiled`
/// partitions the M (feature) dimension across rayon tasks. Each task
/// writes to columns `[panel * M_R, (panel + 1) * M_R)` of `pre_acts` for
/// the panels it owns; column ranges are disjoint across tasks and
/// `M_R = 16` aligns to a 64-byte cache line, so there is neither
/// aliasing nor false sharing. We cannot express that via `&mut [f32]`
/// (rayon's safe APIs split slices by leading dimension, not stride), so
/// we hand the pointer to each task via [`AtomicPtr`], which is
/// `Send + Sync` by construction and dodges Rust 2021's disjoint-capture
/// trap that splits a `*mut f32` newtype back into its non-`Send` field.
use std::sync::atomic::{AtomicPtr, Ordering};

/// Compute a partial tile via scalar arithmetic. Used for:
/// - the trailing batch rows (`batch % N_R != 0`) of every panel
/// - the entire trailing panel (`d_sae % M_R != 0`)
///
/// Per-output arithmetic matches `kernels::scalar::encode_f32` exactly
/// (bias-init + left-to-right K reduction), so this code path is bit-equal
/// to the legacy scalar — only the fast SIMD path introduces the FMA
/// reordering that drives the 1e-5 tolerance.
///
/// # Safety
/// `pre_acts_ptr` must be valid for writes at `b * d_sae + f` for every
/// `b` in `batch_range` and `f` in `[panel * m_r, (panel * m_r + m_r).min(d_sae))`.
/// Concurrent calls with disjoint `(panel, batch_range)` tuples are safe.
unsafe fn scalar_tile_for_panel_raw(
    enc: &EncoderWeights,
    x: &[f32],
    pre_acts_ptr: *mut f32,
    batch_range: std::ops::Range<usize>,
    panel: usize,
    m_r: usize,
) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let f0 = panel * m_r;
    let panel_end = (f0 + m_r).min(d_sae);
    let panel_base = panel * d_in * m_r;
    for b in batch_range {
        for lane in 0..(panel_end - f0) {
            let f = f0 + lane;
            let bias = enc.b_enc[f];
            let mut acc = bias;
            for k in 0..d_in {
                acc += x[b * d_in + k] * enc.w_enc[panel_base + k * m_r + lane];
            }
            *pre_acts_ptr.add(b * d_sae + f) = acc;
        }
    }
}

// ---------------- per-panel helpers (raw-ptr, thread-safe via disjoint cols) ----------------

/// Run the AVX2 microkernel for every full N_R-batch tile on a single
/// full-width panel, then a scalar tail for the partial batch rows on the
/// same panel.
///
/// # Safety
/// Caller ensures `lanes_used == M_R_AVX2` for this panel and that
/// `pre_acts_ptr` is valid for writes at `b * d_sae + f` for every
/// `b ∈ 0..batch`, `f ∈ [panel * M_R_AVX2, (panel + 1) * M_R_AVX2)`.
/// AVX2+FMA target feature must be available (gated at backend select).
unsafe fn process_full_panel_avx2(
    panel: usize,
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts_ptr: *mut f32,
    batch: usize,
) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let m_r = M_R_AVX2;
    let n_full_batches = batch / N_R_AVX2;
    let b_tail_start = n_full_batches * N_R_AVX2;
    let panel_base = panel * d_in * m_r;
    let f0 = panel * m_r;

    let packed_w_panel = enc.w_enc.as_ptr().add(panel_base);
    let bias_base = enc.b_enc.as_ptr().add(f0);

    for batch_block in 0..n_full_batches {
        let b0 = batch_block * N_R_AVX2;
        microkernel_16x6(
            packed_w_panel,
            x.as_ptr().add(b0 * d_in),
            d_in,
            pre_acts_ptr.add(b0 * d_sae + f0),
            d_sae,
            bias_base,
        );
    }
    if b_tail_start < batch {
        scalar_tile_for_panel_raw(enc, x, pre_acts_ptr, b_tail_start..batch, panel, m_r);
    }
}

/// AVX-512 variant of [`process_full_panel_avx2`]. Same SAFETY contract.
unsafe fn process_full_panel_avx512(
    panel: usize,
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts_ptr: *mut f32,
    batch: usize,
) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let m_r = M_R_AVX512;
    let n_full_batches = batch / N_R_AVX512;
    let b_tail_start = n_full_batches * N_R_AVX512;
    let panel_base = panel * d_in * m_r;
    let f0 = panel * m_r;

    let packed_w_panel = enc.w_enc.as_ptr().add(panel_base);
    let bias_base = enc.b_enc.as_ptr().add(f0);

    for batch_block in 0..n_full_batches {
        let b0 = batch_block * N_R_AVX512;
        microkernel_16x12_avx512(
            packed_w_panel,
            x.as_ptr().add(b0 * d_in),
            d_in,
            pre_acts_ptr.add(b0 * d_sae + f0),
            d_sae,
            bias_base,
        );
    }
    if b_tail_start < batch {
        scalar_tile_for_panel_raw(enc, x, pre_acts_ptr, b_tail_start..batch, panel, m_r);
    }
}

// ---------------- AVX2 outer loop ----------------

fn encode_f32_avx2_tiled(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    // SAFETY for the unsafe blocks below: AVX2+FMA detection happens in
    // `select_backend` before this fn pointer is returned; the microkernel
    // is `#[target_feature(enable = "avx2,fma")]`.
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let m_r = match enc.layout {
        WeightLayout::PackedPanels { m_r } => m_r,
        WeightLayout::RowMajor => panic!(
            "avx2_tiled backend called with RowMajor weights — pack via \
             kernels::gemm::pack::repack first",
        ),
    };
    assert_eq!(
        m_r, M_R_AVX2,
        "avx2_tiled expects m_r = {M_R_AVX2}, got {m_r}",
    );
    debug_assert_eq!(x.len(), batch * d_in);
    debug_assert_eq!(pre_acts.len(), batch * d_sae);
    debug_assert_eq!(enc.w_enc.len(), packed_len(d_sae, d_in, m_r));

    let n_panels = panel_count(d_sae, m_r);
    let n_full_panels = d_sae / m_r;
    let has_partial_panel = n_panels > n_full_panels;

    if crate::kernels::encoder::parallel_enabled() && n_full_panels > 1 {
        // M-block grouping: with DEFAULT_M_C = 512 and m_r = 16 we have
        // 32 full panels per M-block — each rayon task processes one
        // M-block's worth of panels so it can amortize task-setup overhead
        // across multiple microkernel invocations. At d_sae = 16384 we
        // generate ~32 tasks for 24 threads, plenty of work-stealing slack
        // for Raptor Lake's heterogeneous P/E cores.
        let panels_per_m_block = (super::DEFAULT_M_C / m_r).max(1);
        let n_m_blocks = n_full_panels.div_ceil(panels_per_m_block);
        let pre_acts_ptr = AtomicPtr::new(pre_acts.as_mut_ptr());

        use rayon::prelude::*;
        (0..n_m_blocks).into_par_iter().for_each(|m_block| {
            let ptr = pre_acts_ptr.load(Ordering::Relaxed);
            let panel_start = m_block * panels_per_m_block;
            let panel_end = ((m_block + 1) * panels_per_m_block).min(n_full_panels);
            for panel in panel_start..panel_end {
                // SAFETY: panel ∈ [0, n_full_panels), so f0 + M_R ≤ d_sae;
                // each task writes columns [f0, f0+M_R) of pre_acts, and
                // disjoint panels across tasks → disjoint column ranges,
                // both aligned to the M_R-float (= 64 B) cache line.
                unsafe {
                    process_full_panel_avx2(panel, x, enc, ptr, batch);
                }
            }
        });
    } else {
        let ptr = pre_acts.as_mut_ptr();
        for panel in 0..n_full_panels {
            unsafe {
                process_full_panel_avx2(panel, x, enc, ptr, batch);
            }
        }
    }

    if has_partial_panel {
        // Partial trailing panel — column range is disjoint from every
        // full panel handled above, so writing it serially after the join
        // is safe. Use the safe slice API (we hold exclusive &mut).
        let panel = n_full_panels;
        let ptr = pre_acts.as_mut_ptr();
        // SAFETY: parallel writes joined; this is the only writer.
        unsafe {
            scalar_tile_for_panel_raw(enc, x, ptr, 0..batch, panel, m_r);
        }
    }
}

pub static AVX2_TILED: Backend = Backend {
    name: "avx2_tiled",
    encode_f32: encode_f32_avx2_tiled,
};

// ---------------- AVX-512 outer loop ----------------

fn encode_f32_avx512_tiled(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    // SAFETY: AVX-512F detection guarded at `select_backend`.
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let m_r = match enc.layout {
        WeightLayout::PackedPanels { m_r } => m_r,
        WeightLayout::RowMajor => panic!(
            "avx512_tiled backend called with RowMajor weights — pack via \
             kernels::gemm::pack::repack first",
        ),
    };
    assert_eq!(
        m_r, M_R_AVX512,
        "avx512_tiled expects m_r = {M_R_AVX512}, got {m_r}",
    );
    debug_assert_eq!(x.len(), batch * d_in);
    debug_assert_eq!(pre_acts.len(), batch * d_sae);
    debug_assert_eq!(enc.w_enc.len(), packed_len(d_sae, d_in, m_r));

    let n_panels = panel_count(d_sae, m_r);
    let n_full_panels = d_sae / m_r;
    let has_partial_panel = n_panels > n_full_panels;

    if crate::kernels::encoder::parallel_enabled() && n_full_panels > 1 {
        let panels_per_m_block = (super::DEFAULT_M_C / m_r).max(1);
        let n_m_blocks = n_full_panels.div_ceil(panels_per_m_block);
        let pre_acts_ptr = AtomicPtr::new(pre_acts.as_mut_ptr());

        use rayon::prelude::*;
        (0..n_m_blocks).into_par_iter().for_each(|m_block| {
            let ptr = pre_acts_ptr.load(Ordering::Relaxed);
            let panel_start = m_block * panels_per_m_block;
            let panel_end = ((m_block + 1) * panels_per_m_block).min(n_full_panels);
            for panel in panel_start..panel_end {
                unsafe {
                    process_full_panel_avx512(panel, x, enc, ptr, batch);
                }
            }
        });
    } else {
        let ptr = pre_acts.as_mut_ptr();
        for panel in 0..n_full_panels {
            unsafe {
                process_full_panel_avx512(panel, x, enc, ptr, batch);
            }
        }
    }

    if has_partial_panel {
        let panel = n_full_panels;
        let ptr = pre_acts.as_mut_ptr();
        unsafe {
            scalar_tile_for_panel_raw(enc, x, ptr, 0..batch, panel, m_r);
        }
    }
}

pub static AVX512_TILED: Backend = Backend {
    name: "avx512_tiled",
    encode_f32: encode_f32_avx512_tiled,
};

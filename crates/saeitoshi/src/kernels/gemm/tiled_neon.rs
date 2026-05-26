//! NEON (16x6) tiled GEMM encoder for aarch64.
//!
//! Same packed-W layout (M_R = 16) as the AVX2 / AVX-512 tiled backends,
//! so the load-time `pack_w_enc` produces a buffer usable by every SIMD
//! tiled backend without re-permutation.
//!
//! Register accounting (32 q-regs, 4 f32 lanes each):
//! - 24 q-regs for accumulators (M_R/4 × N_R = 4 × 6 = 24)
//! - 4  q-regs for the packed-W vector loaded per K
//! - 1  q-reg for the broadcast `x[b, k]` (reused across the 4 FMAs per row)
//! - Total live: 29 q-regs; 3 spare for instruction scheduling slack.
//!
//! Inner loop per K: 4 `vld1q_f32` of W + 6 `vld1q_dup_f32` of x + 24
//! `vfmaq_f32`. On Apple M-series P-cores there are four NEON FMA pipes
//! (4/cycle theoretical), so 24 FMAs take 6 cycles — the inner loop is
//! FMA-bound, not load-bound (4 + 6 = 10 loads vs 4 LSU pipes ≈ 2.5
//! cycles).
//!
//! Verified only via the parity test in CI (`macos-14` runner). Local
//! validation is not possible on the dev box (x86_64 only), so this kernel
//! ships with `1e-5` CI parity as its sole correctness gate until an
//! M-series tester comes online — flagged in `docs/perf-analysis.md`.

#![allow(unsafe_code)]

use std::arch::aarch64::*;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::kernels::Backend;
use crate::sae::{EncoderWeights, WeightLayout};

use super::pack::{packed_len, panel_count};

const M_R_NEON: usize = 16;
const N_R_NEON: usize = 6;

#[target_feature(enable = "neon")]
unsafe fn microkernel_16x6_neon(
    packed_w_panel: *const f32,
    x_base: *const f32,
    d_in: usize,
    pre_acts_tile: *mut f32,
    out_row_stride: usize,
    bias_base: *const f32,
) {
    // Initialize 24 accumulators with bias. Bias is 16 floats = 4 q-regs.
    let b0 = vld1q_f32(bias_base);
    let b1 = vld1q_f32(bias_base.add(4));
    let b2 = vld1q_f32(bias_base.add(8));
    let b3 = vld1q_f32(bias_base.add(12));

    let mut a00 = b0;
    let mut a01 = b1;
    let mut a02 = b2;
    let mut a03 = b3;
    let mut a10 = b0;
    let mut a11 = b1;
    let mut a12 = b2;
    let mut a13 = b3;
    let mut a20 = b0;
    let mut a21 = b1;
    let mut a22 = b2;
    let mut a23 = b3;
    let mut a30 = b0;
    let mut a31 = b1;
    let mut a32 = b2;
    let mut a33 = b3;
    let mut a40 = b0;
    let mut a41 = b1;
    let mut a42 = b2;
    let mut a43 = b3;
    let mut a50 = b0;
    let mut a51 = b1;
    let mut a52 = b2;
    let mut a53 = b3;

    for k in 0..d_in {
        let w_base = packed_w_panel.add(k * M_R_NEON);
        let w0 = vld1q_f32(w_base);
        let w1 = vld1q_f32(w_base.add(4));
        let w2 = vld1q_f32(w_base.add(8));
        let w3 = vld1q_f32(w_base.add(12));

        let x0 = vld1q_dup_f32(x_base.add(k));
        a00 = vfmaq_f32(a00, x0, w0);
        a01 = vfmaq_f32(a01, x0, w1);
        a02 = vfmaq_f32(a02, x0, w2);
        a03 = vfmaq_f32(a03, x0, w3);

        let x1 = vld1q_dup_f32(x_base.add(d_in + k));
        a10 = vfmaq_f32(a10, x1, w0);
        a11 = vfmaq_f32(a11, x1, w1);
        a12 = vfmaq_f32(a12, x1, w2);
        a13 = vfmaq_f32(a13, x1, w3);

        let x2 = vld1q_dup_f32(x_base.add(2 * d_in + k));
        a20 = vfmaq_f32(a20, x2, w0);
        a21 = vfmaq_f32(a21, x2, w1);
        a22 = vfmaq_f32(a22, x2, w2);
        a23 = vfmaq_f32(a23, x2, w3);

        let x3 = vld1q_dup_f32(x_base.add(3 * d_in + k));
        a30 = vfmaq_f32(a30, x3, w0);
        a31 = vfmaq_f32(a31, x3, w1);
        a32 = vfmaq_f32(a32, x3, w2);
        a33 = vfmaq_f32(a33, x3, w3);

        let x4 = vld1q_dup_f32(x_base.add(4 * d_in + k));
        a40 = vfmaq_f32(a40, x4, w0);
        a41 = vfmaq_f32(a41, x4, w1);
        a42 = vfmaq_f32(a42, x4, w2);
        a43 = vfmaq_f32(a43, x4, w3);

        let x5 = vld1q_dup_f32(x_base.add(5 * d_in + k));
        a50 = vfmaq_f32(a50, x5, w0);
        a51 = vfmaq_f32(a51, x5, w1);
        a52 = vfmaq_f32(a52, x5, w2);
        a53 = vfmaq_f32(a53, x5, w3);
    }

    vst1q_f32(pre_acts_tile, a00);
    vst1q_f32(pre_acts_tile.add(4), a01);
    vst1q_f32(pre_acts_tile.add(8), a02);
    vst1q_f32(pre_acts_tile.add(12), a03);
    vst1q_f32(pre_acts_tile.add(out_row_stride), a10);
    vst1q_f32(pre_acts_tile.add(out_row_stride + 4), a11);
    vst1q_f32(pre_acts_tile.add(out_row_stride + 8), a12);
    vst1q_f32(pre_acts_tile.add(out_row_stride + 12), a13);
    vst1q_f32(pre_acts_tile.add(2 * out_row_stride), a20);
    vst1q_f32(pre_acts_tile.add(2 * out_row_stride + 4), a21);
    vst1q_f32(pre_acts_tile.add(2 * out_row_stride + 8), a22);
    vst1q_f32(pre_acts_tile.add(2 * out_row_stride + 12), a23);
    vst1q_f32(pre_acts_tile.add(3 * out_row_stride), a30);
    vst1q_f32(pre_acts_tile.add(3 * out_row_stride + 4), a31);
    vst1q_f32(pre_acts_tile.add(3 * out_row_stride + 8), a32);
    vst1q_f32(pre_acts_tile.add(3 * out_row_stride + 12), a33);
    vst1q_f32(pre_acts_tile.add(4 * out_row_stride), a40);
    vst1q_f32(pre_acts_tile.add(4 * out_row_stride + 4), a41);
    vst1q_f32(pre_acts_tile.add(4 * out_row_stride + 8), a42);
    vst1q_f32(pre_acts_tile.add(4 * out_row_stride + 12), a43);
    vst1q_f32(pre_acts_tile.add(5 * out_row_stride), a50);
    vst1q_f32(pre_acts_tile.add(5 * out_row_stride + 4), a51);
    vst1q_f32(pre_acts_tile.add(5 * out_row_stride + 8), a52);
    vst1q_f32(pre_acts_tile.add(5 * out_row_stride + 12), a53);
}

/// Scalar fallback for partial tiles (tail batch rows + partial trailing
/// panel). Same shape as the x86 version — bit-exact match to the
/// row-major scalar reference modulo packed-vs-row-major address formula.
///
/// # Safety
/// Caller ensures `pre_acts_ptr` is valid for the implied writes and
/// concurrent invocations target disjoint (panel, batch_range) tuples.
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

/// # Safety
/// Caller ensures `lanes_used == M_R_NEON` for this panel, and
/// `pre_acts_ptr` is valid for writes at `b * d_sae + f` for every
/// `b ∈ 0..batch`, `f ∈ [panel * M_R_NEON, (panel + 1) * M_R_NEON)`.
unsafe fn process_full_panel_neon(
    panel: usize,
    x: &[f32],
    enc: &EncoderWeights,
    pre_acts_ptr: *mut f32,
    batch: usize,
) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let m_r = M_R_NEON;
    let n_full_batches = batch / N_R_NEON;
    let b_tail_start = n_full_batches * N_R_NEON;
    let panel_base = panel * d_in * m_r;
    let f0 = panel * m_r;

    let packed_w_panel = enc.w_enc.as_ptr().add(panel_base);
    let bias_base = enc.b_enc.as_ptr().add(f0);

    for batch_block in 0..n_full_batches {
        let b0 = batch_block * N_R_NEON;
        microkernel_16x6_neon(
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

fn encode_f32_neon_tiled(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let m_r = match enc.layout {
        WeightLayout::PackedPanels { m_r } => m_r,
        WeightLayout::RowMajor => panic!(
            "neon_tiled backend called with RowMajor weights — pack via \
             kernels::gemm::pack::repack first",
        ),
    };
    assert_eq!(
        m_r, M_R_NEON,
        "neon_tiled expects m_r = {M_R_NEON}, got {m_r}",
    );
    debug_assert_eq!(x.len(), batch * d_in);
    debug_assert_eq!(pre_acts.len(), batch * d_sae);
    debug_assert_eq!(enc.w_enc.len(), packed_len(d_sae, d_in, m_r));

    let n_panels = panel_count(d_sae, m_r);
    let n_full_panels = d_sae / m_r;
    let has_partial_panel = n_panels > n_full_panels;

    if crate::kernels::encoder::parallel_enabled() && n_full_panels > 1 {
        let panels_per_m_block = (super::m_c_runtime() / m_r).max(1);
        let n_m_blocks = n_full_panels.div_ceil(panels_per_m_block);
        let pre_acts_ptr = AtomicPtr::new(pre_acts.as_mut_ptr());

        use rayon::prelude::*;
        (0..n_m_blocks).into_par_iter().for_each(|m_block| {
            let ptr = pre_acts_ptr.load(Ordering::Relaxed);
            let panel_start = m_block * panels_per_m_block;
            let panel_end = ((m_block + 1) * panels_per_m_block).min(n_full_panels);
            for panel in panel_start..panel_end {
                // SAFETY: see PreActsMut-equivalent argument in tiled_x86 —
                // disjoint cache-line-aligned column ranges per task.
                unsafe {
                    process_full_panel_neon(panel, x, enc, ptr, batch);
                }
            }
        });
    } else {
        let ptr = pre_acts.as_mut_ptr();
        for panel in 0..n_full_panels {
            unsafe {
                process_full_panel_neon(panel, x, enc, ptr, batch);
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

pub static NEON_TILED: Backend = Backend {
    name: "neon_tiled",
    encode_f32: encode_f32_neon_tiled,
};

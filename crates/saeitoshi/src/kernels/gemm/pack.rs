//! W_enc packing helpers for the tiled GEMM backends.
//!
//! The microkernel wants `M_R` weights for the same `k` index loaded as a
//! single contiguous vector (one `vmovups` or `vld1q_f32`). Row-major
//! `[d_sae, d_in]` storage gives us `M_R` features at `M_R` different
//! addresses with stride `d_in * 4 B` — useless for SIMD loads.
//!
//! Packing transposes the local `M_R x K` slab into `K x M_R` panels so
//! the kernel's inner loop streams `M_R` contiguous floats per K step.
//! Because W_enc is loaded once and reused across every encode call, we
//! repack at SAE construction (M4) and amortize the cost forever.
//!
//! Packed layout (for a panel of M_R rows starting at feature `f0`):
//! ```text
//! offset(packed, f0..f0+M_R, k) = panel(f0) * d_in * M_R + k * M_R + lane
//! where panel(f0) = f0 / M_R
//!       lane     = f - f0
//! ```
//!
//! Trailing features (`d_sae % M_R != 0`) are zero-padded in the last
//! panel so the microkernel can read `M_R` lanes branch-free. `b_enc`
//! is NOT padded; the kernel skips lanes that fall past `d_sae` when
//! writing `pre_acts`.

use crate::sae::{EncoderWeights, WeightLayout};

/// Number of M_R-row panels needed to cover `d_sae` features.
///
/// Equivalent to `ceil(d_sae / m_r)` — the last panel may be partially
/// filled with zero-padded lanes (see [`pack_w_enc`]).
#[inline]
pub fn panel_count(d_sae: usize, m_r: usize) -> usize {
    debug_assert!(m_r > 0);
    d_sae.div_ceil(m_r)
}

/// Total length in `f32`s of the packed buffer for a given shape.
#[inline]
pub fn packed_len(d_sae: usize, d_in: usize, m_r: usize) -> usize {
    panel_count(d_sae, m_r) * m_r * d_in
}

/// Repack `w_row_major` (shape `[d_sae, d_in]`) into M_R-row panels.
///
/// Output length is [`packed_len`] floats — slightly larger than the input
/// when `d_sae % m_r != 0` because the tail panel is zero-padded.
///
/// This is a straight permutation: no FP arithmetic, so the packed
/// representation is bit-exact equivalent to the row-major form once
/// re-indexed via the formula in the module doc.
pub fn pack_w_enc(w_row_major: &[f32], d_sae: usize, d_in: usize, m_r: usize) -> Box<[f32]> {
    debug_assert_eq!(w_row_major.len(), d_sae * d_in);
    debug_assert!(m_r > 0);

    let n_panels = panel_count(d_sae, m_r);
    let mut out = vec![0.0f32; n_panels * m_r * d_in].into_boxed_slice();

    // Naive triple-loop transpose. Construction-time cost — runs once per
    // SAE load. Optimize later (64x64 sub-block transpose for TLB locality)
    // only if profiling shows it dominates load time for d_sae > 256K.
    for panel in 0..n_panels {
        let f0 = panel * m_r;
        let lanes_used = (f0 + m_r).min(d_sae) - f0;
        let panel_base = panel * m_r * d_in;
        for lane in 0..lanes_used {
            let src_row_base = (f0 + lane) * d_in;
            for k in 0..d_in {
                out[panel_base + k * m_r + lane] = w_row_major[src_row_base + k];
            }
        }
        // Trailing lanes (lane in lanes_used..m_r) stay 0.0 from init.
    }
    out
}

/// Inverse of [`pack_w_enc`]: convert a packed buffer back to row-major
/// `[d_sae, d_in]`. Used by the `.sit` writer and by tests that need to
/// round-trip the canonical form.
pub fn unpack_w_enc(packed: &[f32], d_sae: usize, d_in: usize, m_r: usize) -> Box<[f32]> {
    debug_assert_eq!(packed.len(), packed_len(d_sae, d_in, m_r));
    let mut out = vec![0.0f32; d_sae * d_in].into_boxed_slice();
    for panel in 0..panel_count(d_sae, m_r) {
        let f0 = panel * m_r;
        let lanes_used = (f0 + m_r).min(d_sae) - f0;
        let panel_base = panel * m_r * d_in;
        for lane in 0..lanes_used {
            let dst_row_base = (f0 + lane) * d_in;
            for k in 0..d_in {
                out[dst_row_base + k] = packed[panel_base + k * m_r + lane];
            }
        }
    }
    out
}

/// Replace the row-major weights in `enc` with packed-panel weights and
/// update [`EncoderWeights::layout`] accordingly. Idempotent if already
/// packed at the same `m_r`; panics if already packed at a different `m_r`.
pub fn repack(enc: &mut EncoderWeights, m_r: usize) {
    if let WeightLayout::PackedPanels { m_r: existing } = enc.layout {
        assert_eq!(
            existing, m_r,
            "EncoderWeights already packed at m_r={existing}, cannot repack at m_r={m_r}",
        );
        return;
    }
    let packed = pack_w_enc(&enc.w_enc, enc.d_sae, enc.d_in, m_r);
    enc.w_enc = packed;
    enc.layout = WeightLayout::PackedPanels { m_r };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_floats(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn pack_then_unpack_roundtrip() {
        for &(d_sae, d_in, m_r) in &[
            (16, 8, 16),
            (17, 8, 16), // padded panel
            (32, 16, 16),
            (33, 11, 8),
            (128, 64, 16),
        ] {
            let w = lcg_floats(d_sae * d_in, 1);
            let packed = pack_w_enc(&w, d_sae, d_in, m_r);
            let unpacked = unpack_w_enc(&packed, d_sae, d_in, m_r);
            assert_eq!(w, unpacked.as_ref().to_vec());
        }
    }

    #[test]
    fn packed_indexing_matches_row_major() {
        // The formula `panel(f0) * d_in * m_r + k * m_r + lane` should
        // address the same value as `w_row_major[f * d_in + k]`.
        let d_sae = 35;
        let d_in = 13;
        let m_r = 16;
        let w = lcg_floats(d_sae * d_in, 7);
        let packed = pack_w_enc(&w, d_sae, d_in, m_r);
        for f in 0..d_sae {
            let panel = f / m_r;
            let lane = f % m_r;
            for k in 0..d_in {
                let from_row = w[f * d_in + k];
                let from_packed = packed[panel * d_in * m_r + k * m_r + lane];
                assert_eq!(from_row, from_packed, "f={f} k={k}");
            }
        }
    }

    #[test]
    fn tail_panel_is_zero_padded() {
        // 17 features at m_r = 16 ⇒ 2 panels, second panel has 1 used + 15 padded.
        let d_sae = 17;
        let d_in = 4;
        let m_r = 16;
        let w = lcg_floats(d_sae * d_in, 9);
        let packed = pack_w_enc(&w, d_sae, d_in, m_r);
        let panel_base = d_in * m_r; // panel index 1
        for k in 0..d_in {
            // Lane 0 of panel 1 holds w[16, k]. Lanes 1..16 are padding.
            assert_eq!(packed[panel_base + k * m_r], w[16 * d_in + k]);
            for lane in 1..m_r {
                assert_eq!(packed[panel_base + k * m_r + lane], 0.0);
            }
        }
    }
}

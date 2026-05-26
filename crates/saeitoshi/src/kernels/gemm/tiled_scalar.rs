//! Scalar reference for the tiled GEMM encoder.
//!
//! This backend exercises the packed-panel layout but does the actual FMAs
//! in plain scalar Rust. Its job is to prove the packing format is
//! addressed correctly — once SIMD microkernels land in M3, they must
//! agree with this reference at 1e-5.
//!
//! Numerical contract: bit-exact match against the row-major scalar
//! reference in `kernels::scalar`. We achieve this by computing each
//! `<W_enc[f], x[b]>` as a full d_in-length reduction with the same
//! left-to-right accumulation order, just reading W via the packed
//! address formula instead of the row-major one. No K-chunking, no
//! cross-lane reordering — the FMA chain is identical to the row-major
//! scalar's, so the f32 results are bit-equal.
//!
//! K-chunking is reintroduced in M3's AVX2 microkernel; the parity gate
//! there relaxes to 1e-5 absolute error rather than bit-exact, which the
//! design analysis (docs/perf-analysis.md, "Parity preservation" section)
//! shows is well within the d_in=2048 worst-case ULP bound.

use crate::kernels::Backend;
use crate::sae::{EncoderWeights, WeightLayout};

use super::pack::panel_count;

/// Encoder: same math as `kernels::scalar::encode_f32` but reads W from the
/// packed-panel layout. Requires `enc.layout == WeightLayout::PackedPanels`.
///
/// Loop structure: outer over panels (groups of M_R contiguous features),
/// then over the lanes within a panel, then over batch rows, then the
/// inner K reduction. This is the same f-outer / b-middle / k-inner order
/// as `kernels::scalar::encode_f32`, so per-output FP arithmetic is
/// bit-equal — the only difference is the address used to fetch each W
/// element.
pub fn encode_f32(x: &[f32], enc: &EncoderWeights, pre_acts: &mut [f32], batch: usize) {
    let d_in = enc.d_in;
    let d_sae = enc.d_sae;
    let m_r = match enc.layout {
        WeightLayout::PackedPanels { m_r } => m_r,
        WeightLayout::RowMajor => panic!(
            "scalar_tiled backend called with RowMajor weights — pack via \
             kernels::gemm::pack::repack first",
        ),
    };
    debug_assert!(m_r > 0);
    debug_assert_eq!(x.len(), batch * d_in);
    debug_assert_eq!(pre_acts.len(), batch * d_sae);
    debug_assert_eq!(enc.b_enc.len(), d_sae);
    debug_assert_eq!(enc.w_enc.len(), super::pack::packed_len(d_sae, d_in, m_r));

    let n_panels = panel_count(d_sae, m_r);

    for panel in 0..n_panels {
        let f0 = panel * m_r;
        let panel_end = (f0 + m_r).min(d_sae);
        let panel_base = panel * d_in * m_r;
        // For each used lane in the panel (skip the zero-padded tail).
        for lane in 0..(panel_end - f0) {
            let f = f0 + lane;
            let bias = enc.b_enc[f];
            for b in 0..batch {
                let x_row_base = b * d_in;
                let mut acc = bias;
                for k in 0..d_in {
                    let w = enc.w_enc[panel_base + k * m_r + lane];
                    acc += x[x_row_base + k] * w;
                }
                pre_acts[b * d_sae + f] = acc;
            }
        }
    }
}

/// Backend descriptor. Selected explicitly by tests/benches in M2; wired
/// into `select_backend` once the SIMD microkernels land and the repack
/// is hoisted into `Sae::from_parts` (M4).
pub static SCALAR_TILED: Backend = Backend {
    name: "scalar_tiled",
    encode_f32,
};

//! Sparsifier enum dispatched once per tile.
//!
//! All five variants live here. Hot-path impls land in M1 (scalar) and M3
//! (SIMD-accelerated TopK).

use crate::sae::SparseOut;

/// Scratch buffers reused across tiles in the TopK selection path.
#[derive(Debug, Default)]
pub struct TopKScratch {
    /// `(score, index)` pairs, used by quickselect / heap.
    pub pairs: Vec<(f32, u32)>,
}

impl TopKScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub enum Sparsifier {
    TopK {
        k: u32,
        post_relu: bool,
        /// Per-feature decoder norm rescaling, applied to pre-acts before
        /// TopK selection. `None` if disabled at load.
        rescale_by_decoder_norm: Option<Vec<f32>>,
    },
    JumpReLU {
        /// Per-feature threshold, length = d_sae.
        thresholds: Vec<f32>,
    },
    BatchTopK {
        k_per_token: u32,
    },
    Relu,
    Gated {
        gate_scale: Vec<f32>,
        gate_bias: Vec<f32>,
    },
}

impl Sparsifier {
    /// Apply this sparsifier to a tile of pre-activations and emit sparse
    /// output rows into `out`. `pre_acts` is `[tile_rows * d_sae]` and may
    /// be mutated in place.
    pub fn apply(
        &self,
        _pre_acts: &mut [f32],
        _tile_rows: usize,
        _d_sae: usize,
        _out: &mut SparseOut,
        _scratch: &mut TopKScratch,
    ) {
        unimplemented!("M1: scalar reference impl per variant")
    }
}

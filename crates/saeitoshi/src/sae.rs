//! The central [`Sae`] type and public encode/decode entry points.
//!
//! Implementation lands in M1. This module currently declares the public
//! surface so downstream crates and tests can compile against a stable shape.

use std::path::Path;

use crate::config::SaeConfig;
use crate::error::Result;

/// A loaded Sparse Autoencoder, ready for inference.
pub struct Sae {
    pub(crate) cfg: SaeConfig,
    // Weight storage, sparsifier, and resolved kernel backend are added in M1.
    // Keeping the struct minimal until the loader lands so we don't ossify
    // field layout prematurely.
    _todo: (),
}

/// Sparse encoder output across a batch of inputs.
///
/// Stored CSR-style: `values[row_offsets[i]..row_offsets[i+1]]` are the
/// non-zero feature values for batch row `i`, with corresponding indices in
/// `indices`.
#[derive(Debug, Default)]
pub struct SparseOut {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub row_offsets: Vec<u32>,
    pub d_sae: u32,
}

impl SparseOut {
    pub fn new(d_sae: u32) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            row_offsets: vec![0],
            d_sae,
        }
    }

    pub fn clear(&mut self) {
        self.indices.clear();
        self.values.clear();
        self.row_offsets.clear();
        self.row_offsets.push(0);
    }

    pub fn batch_size(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }
}

impl Sae {
    /// Load an SAE from disk, auto-detecting the format (SAELens dir,
    /// EleutherAI sparsify layered dir, `.sit` file, or raw safetensors).
    ///
    /// Wired up in M1.
    pub fn load(_path: impl AsRef<Path>) -> Result<Self> {
        unimplemented!("M1: loader auto-detection")
    }

    pub fn config(&self) -> &SaeConfig {
        &self.cfg
    }
}

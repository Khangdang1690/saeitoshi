//! The central [`Sae`] type and public encode/decode entry points.

use std::path::Path;

use crate::config::SaeConfig;
use crate::error::{Result, SaeError};
use crate::kernels::{decoder, encoder, select_backend, Backend};
use crate::normalize;
use crate::sparsify::{Sparsifier, TopKScratch};

/// A loaded Sparse Autoencoder, ready for inference.
pub struct Sae {
    cfg: SaeConfig,
    enc: EncoderWeights,
    dec: DecoderWeights,
    sparsifier: Sparsifier,
    backend: &'static Backend,
}

/// Encoder side of an SAE.
///
/// `w_enc` is stored as `[d_sae, d_in]` row-major. Each row is one feature's
/// encoder direction. This is the transpose of the math convention
/// (`pre_acts = x @ W_math` where `W_math: [d_in, d_sae]`), chosen so the
/// scalar inner loop strides through `d_in` contiguously per output feature.
pub struct EncoderWeights {
    pub w_enc: Box<[f32]>,
    pub b_enc: Box<[f32]>,
    pub d_in: usize,
    pub d_sae: usize,
}

/// Decoder side of an SAE.
///
/// `w_dec` is `[d_sae, d_in]` row-major. Each row is one feature's decoder
/// direction — sparse decode gathers k rows by index and accumulates them
/// scaled by their feature values.
pub struct DecoderWeights {
    pub w_dec: Box<[f32]>,
    pub b_dec: Box<[f32]>,
    pub d_in: usize,
    pub d_sae: usize,
}

/// Sparse encoder output across a batch of inputs.
///
/// CSR-style: `indices[row_offsets[i]..row_offsets[i+1]]` are the active
/// feature ids for batch row `i`, with corresponding values in `values`.
#[derive(Debug, Clone)]
pub struct SparseOut {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    /// Cumulative nnz across rows, length = batch_size + 1.
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

    /// Reset to an empty (zero-row) state, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.indices.clear();
        self.values.clear();
        self.row_offsets.clear();
        self.row_offsets.push(0);
    }

    pub fn batch_size(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    pub fn nnz_in_row(&self, row: usize) -> usize {
        (self.row_offsets[row + 1] - self.row_offsets[row]) as usize
    }

    pub fn row_indices(&self, row: usize) -> &[u32] {
        let s = self.row_offsets[row] as usize;
        let e = self.row_offsets[row + 1] as usize;
        &self.indices[s..e]
    }

    pub fn row_values(&self, row: usize) -> &[f32] {
        let s = self.row_offsets[row] as usize;
        let e = self.row_offsets[row + 1] as usize;
        &self.values[s..e]
    }

    /// Finalize the current row after pushing its (index, value) pairs.
    /// Must be called once per row, even if the row has zero nonzeros.
    pub fn finish_row(&mut self) {
        self.row_offsets.push(self.indices.len() as u32);
    }
}

impl Sae {
    /// Load an SAE from disk, auto-detecting the format. See
    /// [`crate::loader::detect_and_load`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        crate::loader::detect_and_load(path.as_ref())
    }

    /// Construct directly from already-loaded parts. Used by loaders and tests.
    pub fn from_parts(
        cfg: SaeConfig,
        enc: EncoderWeights,
        dec: DecoderWeights,
        sparsifier: Sparsifier,
    ) -> Result<Self> {
        if enc.d_in != cfg.d_in || enc.d_sae != cfg.d_sae {
            return Err(SaeError::ShapeMismatch {
                name: "encoder".into(),
                expected: vec![cfg.d_sae, cfg.d_in],
                got: vec![enc.d_sae, enc.d_in],
            });
        }
        if dec.d_in != cfg.d_in || dec.d_sae != cfg.d_sae {
            return Err(SaeError::ShapeMismatch {
                name: "decoder".into(),
                expected: vec![cfg.d_sae, cfg.d_in],
                got: vec![dec.d_sae, dec.d_in],
            });
        }
        if enc.w_enc.len() != cfg.d_sae * cfg.d_in {
            return Err(SaeError::ShapeMismatch {
                name: "w_enc.len".into(),
                expected: vec![cfg.d_sae * cfg.d_in],
                got: vec![enc.w_enc.len()],
            });
        }
        if dec.w_dec.len() != cfg.d_sae * cfg.d_in {
            return Err(SaeError::ShapeMismatch {
                name: "w_dec.len".into(),
                expected: vec![cfg.d_sae * cfg.d_in],
                got: vec![dec.w_dec.len()],
            });
        }
        if enc.b_enc.len() != cfg.d_sae {
            return Err(SaeError::ShapeMismatch {
                name: "b_enc".into(),
                expected: vec![cfg.d_sae],
                got: vec![enc.b_enc.len()],
            });
        }
        if dec.b_dec.len() != cfg.d_in {
            return Err(SaeError::ShapeMismatch {
                name: "b_dec".into(),
                expected: vec![cfg.d_in],
                got: vec![dec.b_dec.len()],
            });
        }
        let backend = select_backend();
        Ok(Self {
            cfg,
            enc,
            dec,
            sparsifier,
            backend,
        })
    }

    /// Override the auto-detected backend (test + benchmark hook).
    pub fn with_backend(mut self, backend: &'static Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Name of the currently selected kernel backend.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name
    }

    pub fn config(&self) -> &SaeConfig {
        &self.cfg
    }

    pub fn d_in(&self) -> usize {
        self.cfg.d_in
    }

    pub fn d_sae(&self) -> usize {
        self.cfg.d_sae
    }

    pub fn encoder(&self) -> &EncoderWeights {
        &self.enc
    }

    pub fn decoder(&self) -> &DecoderWeights {
        &self.dec
    }

    pub fn sparsifier(&self) -> &Sparsifier {
        &self.sparsifier
    }

    /// Encode a batch of activations into sparse features.
    ///
    /// `x` has length `batch * d_in`, row-major. `out` is cleared then filled.
    pub fn encode(&self, x: &[f32], batch: usize, out: &mut SparseOut) -> Result<()> {
        let d_in = self.cfg.d_in;
        let d_sae = self.cfg.d_sae;
        if x.len() != batch * d_in {
            return Err(SaeError::ShapeMismatch {
                name: "x".into(),
                expected: vec![batch * d_in],
                got: vec![x.len()],
            });
        }
        out.clear();
        out.d_sae = d_sae as u32;

        let mut x_buf: Vec<f32> = Vec::new();
        let x_ref: &[f32] = if self.cfg.apply_b_dec_to_input
            || self.cfg.normalize_activations != crate::config::NormalizeMode::None
        {
            x_buf.extend_from_slice(x);
            if self.cfg.normalize_activations != crate::config::NormalizeMode::None {
                normalize::apply_in_place(self.cfg.normalize_activations, &mut x_buf, batch, d_in);
            }
            if self.cfg.apply_b_dec_to_input {
                for b in 0..batch {
                    let row = &mut x_buf[b * d_in..(b + 1) * d_in];
                    for (v, &b_v) in row.iter_mut().zip(self.dec.b_dec.iter()) {
                        *v -= b_v;
                    }
                }
            }
            &x_buf
        } else {
            x
        };

        let mut pre_acts: Vec<f32> = vec![0.0; batch * d_sae];
        encoder::encode_f32(self.backend, x_ref, &self.enc, &mut pre_acts, batch);

        let mut scratch = TopKScratch::new();
        self.sparsifier
            .apply(&mut pre_acts, batch, d_sae, out, &mut scratch);

        Ok(())
    }

    /// Decode sparse features back into the activation space.
    ///
    /// `out` has length `z.batch_size() * d_in`, row-major.
    pub fn decode(&self, z: &SparseOut, out: &mut [f32]) -> Result<()> {
        let batch = z.batch_size();
        let d_in = self.cfg.d_in;
        if out.len() != batch * d_in {
            return Err(SaeError::ShapeMismatch {
                name: "out".into(),
                expected: vec![batch * d_in],
                got: vec![out.len()],
            });
        }
        if z.d_sae as usize != self.cfg.d_sae {
            return Err(SaeError::ShapeMismatch {
                name: "z.d_sae".into(),
                expected: vec![self.cfg.d_sae],
                got: vec![z.d_sae as usize],
            });
        }
        decoder::decode_sparse(z, &self.dec, out);
        Ok(())
    }

    /// Encode then decode in one shot.
    pub fn reconstruct(&self, x: &[f32], batch: usize, out: &mut [f32]) -> Result<()> {
        let mut z = SparseOut::new(self.cfg.d_sae as u32);
        self.encode(x, batch, &mut z)?;
        self.decode(&z, out)
    }

    /// Serialize this SAE to a `.sit` file (FP32 in v0).
    pub fn write_sit(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::sit::write(self, path.as_ref())
    }
}

//! EleutherAI `sparsify` loader.
//!
//! Format:
//! ```text
//! <dir>/
//!   cfg.json              # {d_in, expansion_factor, k, normalize_decoder, ...}
//!   sae.safetensors       # encoder.weight, encoder.bias, W_dec, b_dec
//! ```
//!
//! Differences from SAELens that the loader handles:
//! - `expansion_factor` instead of `d_sae`.
//! - Tensor names `encoder.weight` / `encoder.bias` (vs `W_enc` / `b_enc`).
//! - Always TopK with `k` from config.
//! - No `apply_b_dec_to_input` semantics in cfg; we default to true to
//!   match sparsify's behavior.

use std::path::Path;

use serde::Deserialize;

use crate::config::{Architecture, NormalizeMode, SaeConfig, WeightDtype};
use crate::error::{Result, SaeError};
use crate::io::safetensors::{read_f32_tensor, transpose, SafetensorsFile};
use crate::sae::{DecoderWeights, EncoderWeights, Sae, WeightLayout};
use crate::sparsify::Sparsifier;

#[derive(Debug, Deserialize)]
struct RawCfg {
    d_in: usize,
    expansion_factor: usize,
    /// TopK k. EleutherAI sparsify always uses TopK.
    k: usize,
    /// Whether features can be negative. Defaults to false.
    #[serde(default)]
    signed: bool,
}

pub fn load_dir(dir: &Path) -> Result<Sae> {
    let cfg_path = dir.join("cfg.json");
    let weights_path = pick_weights_file(dir)?;

    let cfg_bytes = std::fs::read(&cfg_path).map_err(|source| SaeError::Io {
        path: Some(cfg_path.clone()),
        source,
    })?;
    let raw: RawCfg = serde_json::from_slice(&cfg_bytes)?;

    let d_in = raw.d_in;
    let d_sae = d_in * raw.expansion_factor;

    let cfg = SaeConfig {
        architecture: Architecture::Topk,
        d_in,
        d_sae,
        weight_dtype: WeightDtype::F32,
        // sparsify subtracts decoder bias from input internally — match that here.
        apply_b_dec_to_input: true,
        normalize_activations: NormalizeMode::None,
        k: Some(raw.k),
        rescale_by_decoder_norm: false,
    };

    let st = SafetensorsFile::open(&weights_path)?;
    let view = st.view()?;

    // sparsify tensor names. Fall back to SAELens names if a checkpoint uses
    // them — be forgiving.
    let (w_enc, w_enc_shape) =
        read_f32_tensor(&view, "encoder.weight").or_else(|_| read_f32_tensor(&view, "W_enc"))?;
    let (b_enc, b_enc_shape) =
        read_f32_tensor(&view, "encoder.bias").or_else(|_| read_f32_tensor(&view, "b_enc"))?;
    let (w_dec, w_dec_shape) = read_f32_tensor(&view, "W_dec")?;
    let (b_dec, b_dec_shape) = read_f32_tensor(&view, "b_dec")?;

    if b_enc_shape != [d_sae] {
        return Err(SaeError::ShapeMismatch {
            name: "encoder.bias".into(),
            expected: vec![d_sae],
            got: b_enc_shape,
        });
    }
    if b_dec_shape != [d_in] {
        return Err(SaeError::ShapeMismatch {
            name: "b_dec".into(),
            expected: vec![d_in],
            got: b_dec_shape,
        });
    }

    let w_enc = match &w_enc_shape[..] {
        [r, c] if *r == d_sae && *c == d_in => w_enc,
        [r, c] if *r == d_in && *c == d_sae => transpose(&w_enc, d_in, d_sae),
        _ => {
            return Err(SaeError::ShapeMismatch {
                name: "encoder.weight".into(),
                expected: vec![d_sae, d_in],
                got: w_enc_shape,
            });
        }
    };
    let w_dec = match &w_dec_shape[..] {
        [r, c] if *r == d_sae && *c == d_in => w_dec,
        [r, c] if *r == d_in && *c == d_sae => transpose(&w_dec, d_in, d_sae),
        _ => {
            return Err(SaeError::ShapeMismatch {
                name: "W_dec".into(),
                expected: vec![d_sae, d_in],
                got: w_dec_shape,
            });
        }
    };

    let sparsifier = Sparsifier::TopK {
        k: raw.k as u32,
        // sparsify's TopK doesn't ReLU pre-acts before selection in the
        // unsigned case; matches SAELens topk default.
        post_relu: !raw.signed,
        rescale: None,
    };

    let enc = EncoderWeights {
        w_enc,
        b_enc,
        d_in,
        d_sae,
        layout: WeightLayout::RowMajor,
    };
    let dec = DecoderWeights {
        w_dec,
        b_dec,
        d_in,
        d_sae,
    };

    Sae::from_parts(cfg, enc, dec, sparsifier)
}

fn pick_weights_file(dir: &Path) -> Result<std::path::PathBuf> {
    for name in [
        "sae.safetensors",
        "sae_weights.safetensors",
        "weights.safetensors",
    ] {
        let p = dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(SaeError::invalid(
        dir,
        "no sae.safetensors / sae_weights.safetensors found",
    ))
}

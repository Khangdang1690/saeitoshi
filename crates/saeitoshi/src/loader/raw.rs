//! Raw safetensors loader. Caller supplies an explicit [`SaeConfig`] and the
//! path to a bare `sae_weights.safetensors`-style file. Used when there's no
//! cfg.json on disk — e.g. when working with a SAE checkpoint that someone
//! handed you as a single file.

use std::path::Path;

use crate::config::{Architecture, SaeConfig};
use crate::error::{Result, SaeError};
use crate::io::safetensors::{read_f32_tensor, transpose, SafetensorsFile};
use crate::sae::{DecoderWeights, EncoderWeights, Sae};
use crate::sparsify::Sparsifier;

pub fn load_safetensors(path: &Path, cfg: SaeConfig) -> Result<Sae> {
    let d_in = cfg.d_in;
    let d_sae = cfg.d_sae;

    let st = SafetensorsFile::open(path)?;
    let view = st.view()?;

    let (w_enc, w_enc_shape) = read_f32_tensor(&view, "W_enc")
        .or_else(|_| read_f32_tensor(&view, "encoder.weight"))?;
    let (b_enc, b_enc_shape) = read_f32_tensor(&view, "b_enc")
        .or_else(|_| read_f32_tensor(&view, "encoder.bias"))?;
    let (w_dec, w_dec_shape) = read_f32_tensor(&view, "W_dec")?;
    let (b_dec, b_dec_shape) = read_f32_tensor(&view, "b_dec")?;

    if b_enc_shape != [d_sae] {
        return Err(SaeError::ShapeMismatch {
            name: "b_enc".into(),
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
                name: "W_enc".into(),
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

    let sparsifier = match cfg.architecture {
        Architecture::Standard => Sparsifier::Relu,
        Architecture::Topk => Sparsifier::TopK {
            k: cfg.k.ok_or_else(|| {
                SaeError::invalid(path, "topk arch requires `k` in SaeConfig")
            })? as u32,
            post_relu: false,
            rescale: None,
        },
        Architecture::Jumprelu | Architecture::BatchTopk => {
            let (threshold, threshold_shape) = read_f32_tensor(&view, "threshold")?;
            if threshold_shape != [d_sae] {
                return Err(SaeError::ShapeMismatch {
                    name: "threshold".into(),
                    expected: vec![d_sae],
                    got: threshold_shape,
                });
            }
            Sparsifier::JumpReLU {
                thresholds: threshold.into_vec(),
            }
        }
        Architecture::Gated => {
            return Err(SaeError::UnknownArchitecture(
                "gated SAEs not supported in v0".into(),
            ));
        }
    };

    let enc = EncoderWeights { w_enc, b_enc, d_in, d_sae };
    let dec = DecoderWeights { w_dec, b_dec, d_in, d_sae };
    Sae::from_parts(cfg, enc, dec, sparsifier)
}

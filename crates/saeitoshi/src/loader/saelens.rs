//! SAELens loader. Reads `cfg.json` + `sae_weights.safetensors` from a
//! directory and produces a [`crate::sae::Sae`].
//!
//! Supports the standard, topk, jumprelu, and batch_topk architectures.
//! Gated SAEs return [`crate::error::SaeError::UnknownArchitecture`] — they
//! land in a later milestone.

use std::path::Path;

use serde::Deserialize;

use crate::config::{Architecture, NormalizeMode, SaeConfig, WeightDtype};
use crate::error::{Result, SaeError};
use crate::io::safetensors::{read_f32_tensor, transpose, SafetensorsFile};
use crate::sae::{DecoderWeights, EncoderWeights, Sae, WeightLayout};
use crate::sparsify::Sparsifier;

/// Raw cfg.json schema from SAELens. Field set kept conservative — extra
/// fields are ignored.
#[derive(Debug, Deserialize)]
struct RawCfg {
    architecture: Option<String>,
    /// SAELens v6 uses `d_in`/`d_sae`; older releases used `d_in`/`d_hidden`.
    d_in: Option<usize>,
    d_sae: Option<usize>,
    d_hidden: Option<usize>,
    expansion_factor: Option<usize>,

    #[serde(default)]
    dtype: Option<String>,
    #[serde(default = "default_true")]
    apply_b_dec_to_input: bool,
    #[serde(default)]
    normalize_activations: Option<String>,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    activation_fn_str: Option<String>,
    #[serde(default)]
    rescale_acts_by_decoder_norm: bool,

    /// activation_fn for older SAELens checkpoints
    #[serde(default)]
    activation_fn: Option<String>,
}

fn default_true() -> bool {
    true
}

pub fn load_dir(dir: &Path) -> Result<Sae> {
    let cfg_path = dir.join("cfg.json");
    let weights_path = pick_weights_file(dir)?;

    let cfg_bytes = std::fs::read(&cfg_path).map_err(|source| SaeError::Io {
        path: Some(cfg_path.clone()),
        source,
    })?;
    let raw: RawCfg = serde_json::from_slice(&cfg_bytes)?;

    let d_in = raw
        .d_in
        .ok_or_else(|| SaeError::invalid(&cfg_path, "missing d_in"))?;
    let d_sae = raw
        .d_sae
        .or(raw.d_hidden)
        .or_else(|| raw.expansion_factor.map(|e| d_in * e))
        .ok_or_else(|| {
            SaeError::invalid(&cfg_path, "missing d_sae / d_hidden / expansion_factor")
        })?;

    let architecture =
        parse_architecture(raw.architecture.as_deref(), raw.activation_fn.as_deref())?;
    let normalize_activations = parse_normalize(raw.normalize_activations.as_deref());
    let weight_dtype = parse_dtype(raw.dtype.as_deref());

    let cfg = SaeConfig {
        architecture,
        d_in,
        d_sae,
        weight_dtype,
        apply_b_dec_to_input: raw.apply_b_dec_to_input,
        normalize_activations,
        k: raw.k,
        rescale_by_decoder_norm: raw.rescale_acts_by_decoder_norm,
    };

    let st = SafetensorsFile::open(&weights_path)?;
    let view = st.view()?;

    let (w_enc, w_enc_shape) = read_f32_tensor(&view, "W_enc")?;
    let (w_dec, w_dec_shape) = read_f32_tensor(&view, "W_dec")?;
    let (b_enc, b_enc_shape) = read_f32_tensor(&view, "b_enc")?;
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

    // Normalize W_enc to internal layout [d_sae, d_in] row-major.
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

    // Optional scaling_factor: fold into W_dec rows so it costs nothing at runtime.
    let mut w_dec = w_dec;
    if let Ok((scaling, scaling_shape)) = read_f32_tensor(&view, "scaling_factor") {
        if scaling_shape == [d_sae] && !is_all_ones(&scaling) {
            let w_dec_vec = w_dec.into_vec();
            let mut w_dec_vec = w_dec_vec;
            for f in 0..d_sae {
                let s = scaling[f];
                let row = &mut w_dec_vec[f * d_in..(f + 1) * d_in];
                for v in row.iter_mut() {
                    *v *= s;
                }
            }
            w_dec = w_dec_vec.into_boxed_slice();
        }
    }

    // Build the sparsifier from the architecture.
    let sparsifier = match architecture {
        Architecture::Standard => Sparsifier::Relu,
        Architecture::Topk => {
            let k = raw
                .k
                .ok_or_else(|| SaeError::invalid(&cfg_path, "topk arch missing k"))?;
            let post_relu = raw
                .activation_fn_str
                .as_deref()
                .map(|s| s.eq_ignore_ascii_case("topk_relu"))
                .unwrap_or(false);
            let rescale = if raw.rescale_acts_by_decoder_norm {
                Some(per_feature_decoder_norm(&w_dec, d_sae, d_in))
            } else {
                None
            };
            Sparsifier::TopK {
                k: k as u32,
                post_relu,
                rescale,
            }
        }
        Architecture::Jumprelu | Architecture::BatchTopk => {
            let (threshold, threshold_shape) = read_f32_tensor(&view, "threshold")
                .or_else(|_| read_f32_tensor(&view, "log_threshold"))?;
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
                "gated SAEs are not implemented in v0".into(),
            ));
        }
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
        "sae_weights.safetensors",
        "sae.safetensors",
        "weights.safetensors",
    ] {
        let p = dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(SaeError::invalid(
        dir,
        "no sae_weights.safetensors or sae.safetensors found",
    ))
}

fn parse_architecture(arch: Option<&str>, activation_fn: Option<&str>) -> Result<Architecture> {
    match arch.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("standard") | None => {
            // No `architecture` field: infer from `activation_fn`.
            match activation_fn.map(|s| s.to_ascii_lowercase()).as_deref() {
                Some("topk") | Some("topk_relu") => Ok(Architecture::Topk),
                Some("jumprelu") => Ok(Architecture::Jumprelu),
                _ => Ok(Architecture::Standard),
            }
        }
        Some("topk") => Ok(Architecture::Topk),
        Some("jumprelu") => Ok(Architecture::Jumprelu),
        Some("batch_topk") | Some("batchtopk") => Ok(Architecture::BatchTopk),
        Some("gated") => Ok(Architecture::Gated),
        Some(other) => Err(SaeError::UnknownArchitecture(other.to_string())),
    }
}

fn parse_normalize(s: Option<&str>) -> NormalizeMode {
    match s.map(|x| x.to_ascii_lowercase()).as_deref() {
        Some("expected_average_only_in") => NormalizeMode::ExpectedAverageOnlyIn,
        Some("constant_norm_rescale") => NormalizeMode::ConstantNormRescale,
        Some("layer_norm") => NormalizeMode::LayerNorm,
        _ => NormalizeMode::None,
    }
}

fn parse_dtype(s: Option<&str>) -> WeightDtype {
    match s.map(|x| x.to_ascii_lowercase()).as_deref() {
        Some("float16") | Some("f16") => WeightDtype::F16,
        Some("bfloat16") | Some("bf16") => WeightDtype::Bf16,
        Some("int8") | Some("i8") => WeightDtype::I8,
        _ => WeightDtype::F32,
    }
}

fn is_all_ones(v: &[f32]) -> bool {
    v.iter().all(|&x| (x - 1.0).abs() < 1e-7)
}

fn per_feature_decoder_norm(w_dec: &[f32], d_sae: usize, d_in: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(d_sae);
    for f in 0..d_sae {
        let row = &w_dec[f * d_in..(f + 1) * d_in];
        let n: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        out.push(n);
    }
    out
}

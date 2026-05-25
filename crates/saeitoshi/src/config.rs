//! SAE configuration types.
//!
//! `SaeConfig` is the deserialized form of `cfg.json` (SAELens) or the
//! equivalent config in EleutherAI sparsify / our `.sit` format.

use serde::{Deserialize, Serialize};

/// Sparsification architecture. The string values match SAELens's
/// `architecture` field in `cfg.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    /// Plain ReLU + L1 sparsity penalty (the original SAE).
    Standard,
    /// TopK selection over pre-activations (OpenAI / Gao et al. 2024).
    Topk,
    /// JumpReLU with per-feature thresholds (DeepMind / Rajamanoharan et al. 2024).
    Jumprelu,
    /// BatchTopK (Bussmann et al. 2024). Deploys as JumpReLU.
    BatchTopk,
    /// Gated SAE (factored gate × magnitude path).
    Gated,
}

/// On-disk activation-normalization mode applied before the encoder matmul.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizeMode {
    #[default]
    None,
    ExpectedAverageOnlyIn,
    ConstantNormRescale,
    LayerNorm,
}

/// Storage dtype for weight matrices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightDtype {
    #[default]
    F32,
    F16,
    Bf16,
    I8,
}

/// The full SAE configuration, normalized across loader formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaeConfig {
    pub architecture: Architecture,
    pub d_in: usize,
    pub d_sae: usize,
    pub weight_dtype: WeightDtype,
    /// SAELens's `apply_b_dec_to_input`. Subtract `b_dec` from `x` before the
    /// encoder matmul.
    #[serde(default = "default_true")]
    pub apply_b_dec_to_input: bool,
    #[serde(default)]
    pub normalize_activations: NormalizeMode,
    /// For TopK: the k. Ignored for other architectures.
    #[serde(default)]
    pub k: Option<usize>,
    /// For TopK with `rescale_acts_by_decoder_norm = true`, the per-feature
    /// decoder norms baked in at load time.
    #[serde(default)]
    pub rescale_by_decoder_norm: bool,
}

fn default_true() -> bool {
    true
}

impl SaeConfig {
    /// Minimal config for tests / construction from explicit fields.
    pub fn new(architecture: Architecture, d_in: usize, d_sae: usize) -> Self {
        Self {
            architecture,
            d_in,
            d_sae,
            weight_dtype: WeightDtype::F32,
            apply_b_dec_to_input: true,
            normalize_activations: NormalizeMode::None,
            k: None,
            rescale_by_decoder_norm: false,
        }
    }
}

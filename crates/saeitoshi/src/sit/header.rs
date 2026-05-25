//! JSON header schema for `.sit` files. Reserves dtype slots for INT8 / FP16
//! so v0.1 can add quantized weights without a format break.

use serde::{Deserialize, Serialize};

use crate::config::SaeConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SitHeader {
    pub config: SaeConfig,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    /// One of `"f32"`, `"f16"`, `"bf16"`, `"i8"`. v0 only emits `"f32"`.
    pub dtype: String,
    pub shape: Vec<usize>,
    /// File offset, 64-byte aligned.
    pub offset: u64,
    pub nbytes: u64,
    /// For quantized dtypes (v0.1+): name of the companion scale tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_tensor: Option<String>,
}

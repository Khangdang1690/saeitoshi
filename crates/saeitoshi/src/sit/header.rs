//! JSON header schema for `.sit` files.
//!
//! `tensor.offset` is **relative to the start of the tensor section** (which
//! begins at file offset `16 + header_len`, padded up to a 64-byte boundary).
//! Readers compute the absolute file offset as
//! `tensor_section_start + entry.offset`.

use serde::{Deserialize, Serialize};

use crate::config::SaeConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SitHeader {
    pub config: SaeConfig,
    pub sparsifier: SparsifierKind,
    pub tensors: Vec<TensorEntry>,
}

/// Serializable shape of the Sparsifier. Numeric arrays (threshold,
/// rescale) live as tensors named in the manifest, not in the JSON.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SparsifierKind {
    Topk {
        k: u32,
        post_relu: bool,
        has_rescale: bool,
    },
    Jumprelu,
    Relu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    /// One of `"f32"`, `"f16"`, `"bf16"`, `"i8"`. v0 emits `"f32"` only.
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Offset from the start of the tensor section, 64-byte aligned.
    pub offset: u64,
    pub nbytes: u64,
    /// For quantized dtypes (v0.1+): name of the companion scale tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_tensor: Option<String>,
}

//! Checkpoint loaders. Format auto-detected from filesystem layout.

pub mod raw;
pub mod saelens;
pub mod sparsify;

use std::path::Path;

use crate::error::{Result, SaeError};
use crate::sae::Sae;

/// Probe `path` and pick a loader. v0 supports SAELens directory layout
/// (`cfg.json` + `sae_weights.safetensors`). EleutherAI sparsify and raw
/// safetensors land in M5.
pub fn detect_and_load(path: &Path) -> Result<Sae> {
    if path.is_dir() {
        if path.join("cfg.json").exists() {
            return saelens::load_dir(path);
        }
        return Err(SaeError::invalid(
            path,
            "directory does not contain cfg.json",
        ));
    }
    if path.is_file() {
        if path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("json"))
            .unwrap_or(false)
        {
            // Caller passed cfg.json directly — load its parent dir.
            if let Some(parent) = path.parent() {
                return saelens::load_dir(parent);
            }
        }
        return Err(SaeError::invalid(
            path,
            "unrecognized file; pass the SAE directory or a cfg.json",
        ));
    }
    Err(SaeError::invalid(path, "no such path"))
}

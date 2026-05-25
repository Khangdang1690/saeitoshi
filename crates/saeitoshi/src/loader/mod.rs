//! Checkpoint loaders. Format auto-detected from filesystem layout.

pub mod raw;
pub mod saelens;
pub mod sparsify;

use std::path::Path;

use crate::error::{Result, SaeError};
use crate::sae::Sae;

/// Probe `path` and pick a loader. Supports:
/// - SAELens directory (`cfg.json` + `sae_weights.safetensors`).
/// - EleutherAI sparsify directory (`cfg.json` with `expansion_factor` +
///   `sae.safetensors`).
/// - `.sit` single-file (saeitoshi's native format).
///
/// Use [`raw::load_safetensors`] directly when you have a bare
/// `.safetensors` and an explicit [`crate::SaeConfig`].
pub fn detect_and_load(path: &Path) -> Result<Sae> {
    if path.is_dir() {
        let cfg_path = path.join("cfg.json");
        if cfg_path.exists() {
            return dispatch_by_cfg(path, &cfg_path);
        }
        return Err(SaeError::invalid(
            path,
            "directory does not contain cfg.json",
        ));
    }
    if path.is_file() {
        if path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("sit"))
            .unwrap_or(false)
        {
            return crate::sit::read(path);
        }
        if path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("json"))
            .unwrap_or(false)
        {
            // Caller passed cfg.json directly — load its parent dir.
            if let Some(parent) = path.parent() {
                return dispatch_by_cfg(parent, path);
            }
        }
        return Err(SaeError::invalid(
            path,
            "unrecognized file; pass the SAE directory, .sit file, or a cfg.json",
        ));
    }
    Err(SaeError::invalid(path, "no such path"))
}

/// Peek at cfg.json to decide between SAELens and sparsify formats.
fn dispatch_by_cfg(dir: &Path, cfg_path: &Path) -> Result<Sae> {
    let bytes = std::fs::read(cfg_path).map_err(|source| SaeError::Io {
        path: Some(cfg_path.to_path_buf()),
        source,
    })?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    // sparsify cfg has `expansion_factor` and no `d_sae` / `d_hidden`.
    let has_expansion = v.get("expansion_factor").is_some();
    let has_d_sae = v.get("d_sae").is_some() || v.get("d_hidden").is_some();
    if has_expansion && !has_d_sae {
        sparsify::load_dir(dir)
    } else {
        saelens::load_dir(dir)
    }
}

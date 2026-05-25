//! Checkpoint loaders. Format auto-detected from filesystem layout +
//! cfg.json keys.

pub mod raw;
pub mod saelens;
pub mod sparsify;

use std::path::Path;

use crate::error::Result;
use crate::sae::Sae;

/// Loader trait. Each loader knows how to consume one on-disk format and
/// produce a fully-resolved [`Sae`].
pub trait Loader {
    fn load(&self, path: &Path) -> Result<Sae>;
}

/// Probe `path` and pick a loader. Implementation lands in M1.
pub fn detect_and_load(_path: &Path) -> Result<Sae> {
    unimplemented!("M1: format auto-detection")
}

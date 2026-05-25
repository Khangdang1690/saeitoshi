//! Error types.

use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, SaeError>;

#[derive(Debug, Error)]
pub enum SaeError {
    #[error("io error at {path:?}: {source}")]
    Io {
        path: Option<PathBuf>,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid checkpoint at {path}: {reason}")]
    InvalidCheckpoint { path: PathBuf, reason: String },

    #[error("unknown architecture: {0}")]
    UnknownArchitecture(String),

    #[error("missing tensor in checkpoint: {0}")]
    MissingTensor(String),

    #[error("shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl SaeError {
    pub fn invalid(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        SaeError::InvalidCheckpoint {
            path: path.into(),
            reason: reason.into(),
        }
    }
}

impl From<std::io::Error> for SaeError {
    fn from(source: std::io::Error) -> Self {
        SaeError::Io { path: None, source }
    }
}

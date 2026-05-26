//! saeitoshi — fast inference for Sparse Autoencoders.
//!
//! See the crate README for the architectural overview and the published
//! benchmark numbers vs SAELens. This crate is the pure-Rust core; Python
//! bindings live in the `saeitoshi-python` crate.

#![deny(unsafe_code)]

pub mod config;
pub mod dtype;
pub mod error;
pub mod io;
pub mod kernels;
pub mod loader;
pub mod normalize;
pub mod sae;
pub mod sit;
pub mod sparsify;
pub mod stream;
pub mod topk_examples;

pub use config::{Architecture, NormalizeMode, SaeConfig, WeightDtype};
pub use error::{Result, SaeError};
pub use kernels::{select_backend, Backend};
pub use sae::{Sae, SparseOut, WeightLayout};
pub use sparsify::Sparsifier;

/// Backend statics exposed for tests + benchmarks.
pub mod backends {
    pub use crate::kernels::scalar::SCALAR;

    #[cfg(target_arch = "x86_64")]
    pub use crate::kernels::x86::{AVX2, AVX512};

    #[cfg(target_arch = "aarch64")]
    pub use crate::kernels::aarch64::NEON;

    #[cfg(feature = "perf-v2")]
    pub use crate::kernels::gemm::tiled_scalar::SCALAR_TILED;
}

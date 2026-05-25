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
pub use sae::{Sae, SparseOut};
pub use sparsify::Sparsifier;

//! Streaming encode over disk-backed activation stores.
//!
//! Public types (`StreamingEncoder`, `BufferPool`) land in M4.

pub mod mmap;
pub mod source;

pub use source::ActivationSource;

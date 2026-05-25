//! safetensors helpers. We parse the header ourselves and mmap the body,
//! rather than relying on the `safetensors` crate's `mmap` feature (which
//! bypasses on some Windows versions).
//!
//! Implementation lands in M1 alongside the SAELens loader.

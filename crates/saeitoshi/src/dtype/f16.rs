//! FP16 weight storage helpers.
//!
//! Up-cast to f32 inside the SIMD register per accumulator step. Implementation
//! lives in [`crate::kernels::encoder`]; this module is a placeholder for any
//! storage-specific helpers we discover we need.

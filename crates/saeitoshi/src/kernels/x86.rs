//! x86_64 SIMD kernels: AVX2 + AVX-512BW.
//!
//! All intrinsics are wrapped in `#[target_feature(enable = "...")]` fns
//! reachable only via the runtime-detection branch in
//! [`crate::kernels::select_backend`]. Implementation lands in M3.

#![allow(unsafe_code)]

use super::Backend;

pub static AVX2: Backend = Backend { name: "avx2" };
pub static AVX512: Backend = Backend { name: "avx512" };

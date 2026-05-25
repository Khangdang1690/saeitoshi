//! Kernel module: runtime dispatch between scalar / AVX2 / AVX-512 / NEON.
//!
//! Layout:
//! - [`Backend`] holds safe function pointers for each kernel operation.
//!   Pointers are populated to either scalar reference impls or to
//!   `#[target_feature]`-gated SIMD impls. The Sae stores `&'static Backend`
//!   resolved once at construction via [`select_backend`].
//! - All `unsafe` SIMD intrinsics live behind a thin safe wrapper in the
//!   per-arch submodules; the wrapper is what's stored in `Backend`.

use crate::sae::EncoderWeights;

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod decoder;
pub mod encoder;
pub mod topk;

/// Encoder kernel signature. Implementations must be **safe to call** —
/// SIMD impls go through a safe wrapper that asserts the relevant
/// `#[target_feature]` is present (guaranteed by [`select_backend`]).
pub type EncodeFn = fn(&[f32], &EncoderWeights, &mut [f32], usize);

/// Resolved CPU-specific kernel table. One static per backend, addressed
/// as `&'static Backend` everywhere.
pub struct Backend {
    pub name: &'static str,
    pub encode_f32: EncodeFn,
}

/// Pick the fastest backend available on the current CPU.
///
/// Called once per `Sae` at construction. Cheap — `is_x86_feature_detected!`
/// caches the cpuid lookup internally.
pub fn select_backend() -> &'static Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return &x86::AVX512;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return &x86::AVX2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return &aarch64::NEON;
        }
    }
    &scalar::SCALAR
}

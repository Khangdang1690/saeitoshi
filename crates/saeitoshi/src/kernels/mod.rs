//! Kernel module: runtime dispatch over tiled GEMM backends.
//!
//! [`Backend`] holds safe function pointers; each pointer routes to a
//! `#[target_feature]`-gated SIMD microkernel or to the scalar-tiled
//! fallback. The Sae stores `&'static Backend` resolved once at
//! construction via [`select_backend`]. All `unsafe` SIMD intrinsics live
//! behind a thin safe wrapper in the per-arch submodules.

use crate::sae::EncoderWeights;

pub mod decoder;
pub mod encoder;
pub mod gemm;
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

/// Pick the fastest tiled GEMM backend available on the current CPU.
///
/// Called once per `Sae` at construction. Cheap — `is_x86_feature_detected!`
/// caches the cpuid lookup internally.
pub fn select_backend() -> &'static Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return &gemm::x86::AVX512;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return &gemm::x86::AVX2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return &gemm::neon::NEON;
        }
    }
    &gemm::scalar::SCALAR
}

/// Packing `m_r` for the given backend. Every shipping backend is tiled
/// and shares the same packed layout, so this always returns
/// `Some(gemm::DEFAULT_M_R)` — kept as a function so `Sae::from_parts`
/// can stay structurally identical and future per-arch m_r variants slot
/// in without a call-site change.
pub fn backend_m_r(_backend: &Backend) -> Option<usize> {
    Some(gemm::DEFAULT_M_R)
}

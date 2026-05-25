//! Kernel module: runtime dispatch between scalar / AVX2 / AVX-512 / NEON.
//!
//! All `unsafe` SIMD intrinsics are walled off in the per-arch submodules
//! gated by `#[cfg(target_arch = ...)]`. The scalar reference path is
//! the parity gate (matches PyTorch op order).

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod decoder;
pub mod encoder;
pub mod topk;

/// Vtable of the kernels for one runtime-detected CPU configuration.
/// Field types are TBD until the M1 reference path lands; we fill them in
/// then so the kernel signatures stay consistent across backends.
pub struct Backend {
    pub name: &'static str,
}

/// Resolve the best backend for the current CPU. Called once at load time
/// per `Sae` instance.
pub fn select_backend() -> &'static Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") {
            return &x86::AVX512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
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

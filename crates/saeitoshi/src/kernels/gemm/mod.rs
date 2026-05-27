//! Cache-tiled, register-blocked GEMM encoder.
//!
//! See `docs/perf-analysis.md` for the roofline analysis that motivates
//! the BLIS-style 5-loop blocking:
//!
//! ```text
//! for m_block in (0..d_sae).step_by(M_C):       # threads partition here
//!   for n_block in (0..batch).step_by(N_C):
//!     for k_chunk in (0..d_in).step_by(K_C):
//!       for f0 in m_block..m_end step M_R:      # microkernel
//!         for b0 in n_block..n_end step N_R:
//!           microkernel(M_R x N_R tile, K_C k's)
//! ```
//!
//! M is `d_sae` (features, output rows), N is `batch`, K is `d_in`. The
//! microkernel produces an `M_R x N_R` register tile of pre-activations,
//! streaming `K_C` reduction elements through a per-ISA `#[target_feature]`
//! kernel.

pub mod pack;
pub mod tiled_scalar;

#[cfg(target_arch = "x86_64")]
pub mod tiled_x86;

#[cfg(target_arch = "aarch64")]
pub mod tiled_neon;

/// Resolve the M-block grouping size from `SAEITOSHI_GEMM_TILE=MC,NC,KC`,
/// falling back to [`DEFAULT_M_C`].
///
/// Only `MC` is consumed today — `NC` and `KC` are parsed but ignored
/// until the K-tiled variant lands. The env var is parsed once per
/// process at first call; later changes are not picked up.
///
/// Use it from the tiled backends to compute `panels_per_m_block`. For
/// `DEFAULT_M_C = 512, m_r = 16` this gives 32 panels per rayon task on
/// the dev box; an override like `SAEITOSHI_GEMM_TILE=256,8192,512` would
/// halve the task size (useful on smaller-L2 CPUs).
pub fn m_c_runtime() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("SAEITOSHI_GEMM_TILE") {
        Ok(spec) => spec
            .split(',')
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_M_C),
        Err(_) => DEFAULT_M_C,
    })
}

/// Default register-tile row count (M_R).
///
/// Tuned for AVX2: M_R = 16 = 2 ymm registers wide. Same value works for
/// NEON (4 q-regs) and the scalar reference. An AVX-512 32-wide variant
/// would need its own packing pass.
///
/// The packing layout depends on M_R, so it's recorded in the
/// [`crate::sae::WeightLayout::PackedPanels { m_r }`] variant on
/// [`EncoderWeights`](crate::sae::EncoderWeights) and the kernel reads it
/// from there rather than assuming a global constant.
pub const DEFAULT_M_R: usize = 16;

/// Default cache tile sizes, tuned for Intel Raptor Lake (i9-13900HX).
///
/// - **K_C = 512** keeps `(M_R + N_R) * K_C * 4 B = 44 KB` resident in the
///   48 KB P-core L1d. For `d_in <= 512` the kernel runs without K-tiling;
///   `d_in = 2048` (headline) → 4 K-chunks.
/// - **M_C = 512** keeps the packed-A panel `M_C * K_C * 4 B = 1 MB` in
///   the 2 MB private P-core L2 with headroom for other working set.
///   `d_sae = 16384` / `M_C = 512` = 32 M-blocks, a clean divide across
///   24 rayon threads with work-stealing for the trailing tasks.
/// - **N_C** defaults to the actual batch size (batches are small enough
///   to fit in L3 / private caches). The constant below is a cap used
///   only when `batch` is unusually large.
///
/// Apple M-series and AVX-512 server chips get the same defaults for now;
/// `SAEITOSHI_GEMM_TILE=MC,NC,KC` overrides MC per machine.
pub const DEFAULT_K_C: usize = 512;
pub const DEFAULT_M_C: usize = 512;
pub const DEFAULT_N_C_CAP: usize = 8192;

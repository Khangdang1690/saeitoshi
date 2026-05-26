//! perf-v2: cache-tiled, register-blocked GEMM encoder.
//!
//! The legacy backends (`kernels::{scalar, x86, aarch64}`) compute each
//! `<W_enc[f], x[b]>` as an independent scalar reduction. That dot-product
//! pattern is latency-bound at FMA chain depth and — once parallelism
//! enters — duplicates the W DRAM stream across threads. See
//! `docs/perf-analysis.md` for the roofline that motivates this module.
//!
//! Structure (BLIS 5-loop blocking):
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
//!
//! Commit-by-commit progression:
//!   - **Commit 2 (this file)**: scaffold + packing helpers + scalar tiled
//!     reference. No SIMD yet; proves the packing layout is correct against
//!     the row-major scalar reference at bit-exact parity.
//!   - **Commit 3**: AVX2 (16x6) + AVX-512 (32x12) microkernels.
//!   - **Commit 4**: hoist the repack into `Sae::from_parts`, rewrite the
//!     rayon split to M-block partitioning (threads share the x panel via
//!     L3 instead of duplicating W).
//!   - **Commit 5**: NEON (16x8) microkernel + software prefetch +
//!     `SAEITOSHI_GEMM_TILE=MC,NC,KC` env override.
//!   - **Commit 6**: flip default backend.

pub mod pack;
pub mod tiled_scalar;

#[cfg(target_arch = "x86_64")]
pub mod tiled_x86;

/// Default register-tile row count (M_R).
///
/// Tuned for AVX2: M_R = 16 = 2 ymm registers wide. Same value works for
/// NEON (4 q-regs) and the scalar reference. AVX-512 would benefit from
/// M_R = 32 (2 zmm wide); that ISA gets its own packing pass in M3.
///
/// The packing layout depends on M_R, so this is recorded in the
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
/// - **N_C** defaults to the actual batch size for v0.1 (batches are
///   small enough to fit in L3 / private caches). The constant below is
///   a cap used only when `batch` is unusually large.
///
/// Apple M-series and AVX-512 server chips get the same defaults for now;
/// see the env override in M5 for per-machine tuning.
pub const DEFAULT_K_C: usize = 512;
pub const DEFAULT_M_C: usize = 512;
pub const DEFAULT_N_C_CAP: usize = 8192;

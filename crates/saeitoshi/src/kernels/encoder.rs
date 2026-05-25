//! Encoder matmul: `[B, d_in] @ [d_in, d_sae] + b_enc -> [B, d_sae]`.
//!
//! This is the hottest path in the library. Blocked GEMM tile loop, generic
//! over weight dtype and SIMD backend. Implementation lands in M1 (scalar)
//! and M3 (SIMD).

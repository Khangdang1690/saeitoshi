//! Portable scalar reference kernels.
//!
//! This is the parity gate: scalar kernels must match SAELens (PyTorch CPU)
//! to within 1e-5. SIMD backends in turn must match the scalar reference to
//! within 1e-5 — guaranteeing transitively that SIMD outputs are within
//! ~2e-5 of SAELens (in practice they're tighter once tile sizes are tuned).
//!
//! M1 implements the scalar paths here.

use super::Backend;

pub static SCALAR: Backend = Backend { name: "scalar" };

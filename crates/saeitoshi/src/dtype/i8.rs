//! INT8 weight storage helpers.
//!
//! Dequant happens in SIMD register: `vpmovsxbw` -> `vcvtdq2ps` -> `vmulps`
//! by a broadcast scale. Per-column for the encoder, per-row for the decoder.
//! v0.1 work — we load INT8 weights, we don't produce them at runtime; that
//! lives in `scripts/quantize_saelens.py`.

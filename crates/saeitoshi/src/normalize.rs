//! Pre-encoder activation normalization. Must be bit-exact with SAELens.
//!
//! Implementation lands in M1 alongside the scalar reference kernel.

use crate::config::NormalizeMode;

/// Apply the configured normalization to a `[batch * d_in]` activation tile,
/// in place. Returns any per-row scale factors needed to denormalize the
/// reconstruction on the way out.
pub fn apply_in_place(
    _mode: NormalizeMode,
    _x: &mut [f32],
    _batch: usize,
    _d_in: usize,
    _scratch_scales: &mut Vec<f32>,
) {
    unimplemented!("M1: scalar normalize per mode")
}

//! Pre-encoder activation normalization. Bit-exact with SAELens.
//!
//! SAELens supports four modes via `cfg.normalize_activations`:
//! - `None`: no-op.
//! - `ExpectedAverageOnlyIn`: subtract per-row mean (over d_in).
//! - `ConstantNormRescale`: rescale each row so its L2 norm equals sqrt(d_in).
//! - `LayerNorm`: standard layer norm — subtract mean, divide by stddev (eps=1e-5).

use crate::config::NormalizeMode;

const LAYER_NORM_EPS: f32 = 1e-5;

/// Apply the configured normalization to a `[batch * d_in]` activation tile
/// in place.
pub fn apply_in_place(mode: NormalizeMode, x: &mut [f32], batch: usize, d_in: usize) {
    debug_assert_eq!(x.len(), batch * d_in);
    match mode {
        NormalizeMode::None => {}
        NormalizeMode::ExpectedAverageOnlyIn => {
            for b in 0..batch {
                let row = &mut x[b * d_in..(b + 1) * d_in];
                let mean: f32 = row.iter().sum::<f32>() / d_in as f32;
                for v in row.iter_mut() {
                    *v -= mean;
                }
            }
        }
        NormalizeMode::ConstantNormRescale => {
            let target = (d_in as f32).sqrt();
            for b in 0..batch {
                let row = &mut x[b * d_in..(b + 1) * d_in];
                let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    let scale = target / norm;
                    for v in row.iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }
        NormalizeMode::LayerNorm => {
            for b in 0..batch {
                let row = &mut x[b * d_in..(b + 1) * d_in];
                let mean: f32 = row.iter().sum::<f32>() / d_in as f32;
                let var: f32 =
                    row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d_in as f32;
                let inv_std = 1.0 / (var + LAYER_NORM_EPS).sqrt();
                for v in row.iter_mut() {
                    *v = (*v - mean) * inv_std;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_noop() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let orig = x.clone();
        apply_in_place(NormalizeMode::None, &mut x, 1, 4);
        assert_eq!(x, orig);
    }

    #[test]
    fn expected_average_only_in_subtracts_row_mean() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        apply_in_place(NormalizeMode::ExpectedAverageOnlyIn, &mut x, 1, 4);
        let mean: f32 = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        assert!((x[0] - (1.0 - mean)).abs() < 1e-6);
        assert!((x[3] - (4.0 - mean)).abs() < 1e-6);
        // Post-norm row sums to ~0.
        assert!(x.iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn constant_norm_rescale_targets_sqrt_d_in() {
        let mut x = vec![1.0, 2.0, 2.0, 4.0];
        let d_in = 4usize;
        apply_in_place(NormalizeMode::ConstantNormRescale, &mut x, 1, d_in);
        let new_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let target = (d_in as f32).sqrt();
        assert!((new_norm - target).abs() < 1e-5, "{new_norm} vs {target}");
    }

    #[test]
    fn layer_norm_normalizes_to_unit_variance() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        apply_in_place(NormalizeMode::LayerNorm, &mut x, 1, 5);
        let mean: f32 = x.iter().sum::<f32>() / 5.0;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 5.0;
        assert!(mean.abs() < 1e-5);
        // Variance is < 1.0 because of the epsilon; just check it's close.
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn batched_rows_are_independent() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
        apply_in_place(NormalizeMode::ExpectedAverageOnlyIn, &mut x, 2, 4);
        // Each row should sum to ~0.
        assert!(x[..4].iter().sum::<f32>().abs() < 1e-4);
        assert!(x[4..].iter().sum::<f32>().abs() < 1e-3);
    }
}

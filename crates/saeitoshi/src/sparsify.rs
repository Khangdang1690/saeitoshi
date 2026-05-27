//! Sparsifier enum dispatched once per tile.

use crate::kernels::topk;
use crate::sae::SparseOut;

/// Scratch buffers reused across tiles in the TopK selection path.
#[derive(Debug, Default)]
pub struct TopKScratch {
    pub indexed: Vec<(u32, f32)>,
}

impl TopKScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Sparsification strategy applied to pre-activations.
///
/// BatchTopK and Standard (ReLU+L1) are resolved at load time into one
/// of these:
/// - BatchTopK → JumpReLU with the saved per-feature thresholds.
/// - Standard → Relu.
#[derive(Debug, Clone)]
pub enum Sparsifier {
    TopK {
        k: u32,
        post_relu: bool,
        /// Per-feature decoder-norm rescaling baked in at load time.
        /// Multiplied into pre-acts before TopK selection; None if disabled.
        rescale: Option<Vec<f32>>,
    },
    JumpReLU {
        thresholds: Vec<f32>,
    },
    Relu,
}

impl Sparsifier {
    /// Apply this sparsifier to a tile of pre-activations. `pre_acts` is
    /// `[tile_rows * d_sae]` and may be mutated (e.g. by TopK's rescale).
    pub fn apply(
        &self,
        pre_acts: &mut [f32],
        tile_rows: usize,
        d_sae: usize,
        out: &mut SparseOut,
        scratch: &mut TopKScratch,
    ) {
        match self {
            Sparsifier::TopK {
                k,
                post_relu,
                rescale,
            } => {
                apply_topk(
                    pre_acts,
                    tile_rows,
                    d_sae,
                    *k as usize,
                    *post_relu,
                    rescale.as_deref(),
                    out,
                    scratch,
                );
            }
            Sparsifier::JumpReLU { thresholds } => {
                apply_jumprelu(pre_acts, tile_rows, d_sae, thresholds, out);
            }
            Sparsifier::Relu => {
                apply_relu(pre_acts, tile_rows, d_sae, out);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_topk(
    pre_acts: &mut [f32],
    tile_rows: usize,
    d_sae: usize,
    k: usize,
    post_relu: bool,
    rescale: Option<&[f32]>,
    out: &mut SparseOut,
    scratch: &mut TopKScratch,
) {
    let k = k.min(d_sae);

    // Rayon dispatch overhead is ~10us; only parallelize when there's
    // enough work per call to amortize it. Threshold picked so that
    // d_sae=16384 needs tile_rows>=4, d_sae=2048 needs tile_rows>=16.
    const PAR_THRESHOLD_ELEMS: usize = 8 * 1024;
    let use_par = tile_rows >= 4 && tile_rows.saturating_mul(d_sae) >= PAR_THRESHOLD_ELEMS;

    if !use_par {
        for r in 0..tile_rows {
            let row = &mut pre_acts[r * d_sae..(r + 1) * d_sae];
            if let Some(scales) = rescale {
                for (v, &s) in row.iter_mut().zip(scales.iter()) {
                    *v *= s;
                }
            }
            topk::topk_select(row, k, scratch);
            for &(idx, val) in scratch.indexed.iter() {
                let v = if post_relu { val.max(0.0) } else { val };
                if post_relu && v == 0.0 {
                    continue;
                }
                out.indices.push(idx);
                out.values.push(v);
            }
            out.finish_row();
        }
        return;
    }

    // Parallel path: each row writes into pre-sized slots in row_idx /
    // row_val (capacity k per row). row_lens tracks the actual count
    // after post_relu drops. A serial compaction pass then extends the
    // CSR in row order.
    use rayon::prelude::*;

    let mut row_idx: Vec<u32> = vec![0; tile_rows * k];
    let mut row_val: Vec<f32> = vec![0.0; tile_rows * k];
    let mut row_lens: Vec<u32> = vec![0; tile_rows];

    pre_acts
        .par_chunks_mut(d_sae)
        .zip(row_idx.par_chunks_mut(k.max(1)))
        .zip(row_val.par_chunks_mut(k.max(1)))
        .zip(row_lens.par_iter_mut())
        .for_each_init(
            TopKScratch::new,
            |thread_scratch, (((row, idx_slot), val_slot), len)| {
                if let Some(scales) = rescale {
                    for (v, &s) in row.iter_mut().zip(scales.iter()) {
                        *v *= s;
                    }
                }
                topk::topk_select(row, k, thread_scratch);
                let mut n: u32 = 0;
                for &(idx, val) in thread_scratch.indexed.iter() {
                    let v = if post_relu { val.max(0.0) } else { val };
                    if post_relu && v == 0.0 {
                        continue;
                    }
                    idx_slot[n as usize] = idx;
                    val_slot[n as usize] = v;
                    n += 1;
                }
                *len = n;
            },
        );

    let total: usize = row_lens.iter().map(|&n| n as usize).sum();
    out.indices.reserve(total);
    out.values.reserve(total);
    for (r, &len) in row_lens.iter().enumerate() {
        let n = len as usize;
        let start = r * k;
        out.indices.extend_from_slice(&row_idx[start..start + n]);
        out.values.extend_from_slice(&row_val[start..start + n]);
        out.finish_row();
    }
}

fn apply_jumprelu(
    pre_acts: &[f32],
    tile_rows: usize,
    d_sae: usize,
    thresholds: &[f32],
    out: &mut SparseOut,
) {
    debug_assert_eq!(thresholds.len(), d_sae);
    // SAELens JumpReLU: (x > threshold) * ReLU(x) = max(x, 0) where x > threshold,
    // else 0. We use `>` (strict) to match SAELens exactly.
    for r in 0..tile_rows {
        let row = &pre_acts[r * d_sae..(r + 1) * d_sae];
        for (f, (&v, &t)) in row.iter().zip(thresholds.iter()).enumerate() {
            if v > t {
                let out_v = v.max(0.0);
                if out_v > 0.0 {
                    out.indices.push(f as u32);
                    out.values.push(out_v);
                }
            }
        }
        out.finish_row();
    }
}

fn apply_relu(pre_acts: &[f32], tile_rows: usize, d_sae: usize, out: &mut SparseOut) {
    for r in 0..tile_rows {
        let row = &pre_acts[r * d_sae..(r + 1) * d_sae];
        for (f, &v) in row.iter().enumerate() {
            if v > 0.0 {
                out.indices.push(f as u32);
                out.values.push(v);
            }
        }
        out.finish_row();
    }
}

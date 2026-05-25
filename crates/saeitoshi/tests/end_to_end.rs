//! End-to-end correctness tests for the scalar reference path.
//!
//! These tests build SAEs in memory with hand-crafted weights, run them
//! through the public API, and assert mathematical invariants:
//!   - Identity-like SAE round-trips a known vector to within float rounding.
//!   - JumpReLU output zeros entries at-or-below threshold.
//!   - TopK output preserves exactly k entries with stable tie-breaking.
//!   - SAELens loader recovers the same SAE we wrote to disk.

use std::collections::BTreeMap;

use safetensors::{serialize_to_file, tensor::TensorView, Dtype};

use saeitoshi::config::{Architecture, NormalizeMode, SaeConfig, WeightDtype};
use saeitoshi::sae::{DecoderWeights, EncoderWeights, Sae, SparseOut};
use saeitoshi::sparsify::Sparsifier;

fn identity_sae(d: usize, k: usize) -> Sae {
    // W_enc and W_dec are identity matrices; b_enc, b_dec are zero.
    // TopK with k=d selects everything → exact round-trip on positive inputs.
    let mut w_enc = vec![0.0f32; d * d];
    let mut w_dec = vec![0.0f32; d * d];
    for f in 0..d {
        w_enc[f * d + f] = 1.0;
        w_dec[f * d + f] = 1.0;
    }
    let enc = EncoderWeights {
        w_enc: w_enc.into_boxed_slice(),
        b_enc: vec![0.0; d].into_boxed_slice(),
        d_in: d,
        d_sae: d,
    };
    let dec = DecoderWeights {
        w_dec: w_dec.into_boxed_slice(),
        b_dec: vec![0.0; d].into_boxed_slice(),
        d_in: d,
        d_sae: d,
    };
    let cfg = SaeConfig {
        architecture: Architecture::Topk,
        d_in: d,
        d_sae: d,
        weight_dtype: WeightDtype::F32,
        apply_b_dec_to_input: false,
        normalize_activations: NormalizeMode::None,
        k: Some(k),
        rescale_by_decoder_norm: false,
    };
    let sparsifier = Sparsifier::TopK {
        k: k as u32,
        post_relu: false,
        rescale: None,
    };
    Sae::from_parts(cfg, enc, dec, sparsifier).unwrap()
}

#[test]
fn identity_sae_roundtrip() {
    let d = 8;
    let sae = identity_sae(d, d);
    let x: Vec<f32> = (1..=d).map(|i| i as f32).collect();
    let mut recon = vec![0.0f32; d];
    sae.reconstruct(&x, 1, &mut recon).unwrap();
    for (a, b) in x.iter().zip(recon.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn topk_selects_k_largest() {
    let d = 8;
    let sae = identity_sae(d, 3);
    let x = vec![1.0f32, 5.0, 2.0, 4.0, 3.0, 0.5, 7.0, 6.0];
    let mut z = SparseOut::new(d as u32);
    sae.encode(&x, 1, &mut z).unwrap();
    assert_eq!(z.batch_size(), 1);
    assert_eq!(z.nnz_in_row(0), 3);
    // top-3 in x are indices {1: 5.0, 6: 7.0, 7: 6.0}; sorted by index ascending.
    assert_eq!(z.row_indices(0), &[1, 6, 7]);
    let vs = z.row_values(0);
    assert!((vs[0] - 5.0).abs() < 1e-6);
    assert!((vs[1] - 7.0).abs() < 1e-6);
    assert!((vs[2] - 6.0).abs() < 1e-6);
}

#[test]
fn topk_tie_break_index_ascending() {
    let d = 4;
    let sae = identity_sae(d, 2);
    // All scores equal — stable tie-break should pick the two lowest indices.
    let x = vec![3.0f32, 3.0, 3.0, 3.0];
    let mut z = SparseOut::new(d as u32);
    sae.encode(&x, 1, &mut z).unwrap();
    assert_eq!(z.row_indices(0), &[0, 1]);
}

#[test]
fn jumprelu_strict_threshold_and_relu_floor() {
    let d = 4;
    let mut sae = identity_sae(d, d);
    // Replace sparsifier with JumpReLU thresholds=[0.5, 0.5, 0.5, 0.5].
    // Use from_parts since fields are private.
    let cfg = sae.config().clone();
    let enc = EncoderWeights {
        w_enc: sae.encoder().w_enc.clone(),
        b_enc: sae.encoder().b_enc.clone(),
        d_in: d,
        d_sae: d,
    };
    let dec = DecoderWeights {
        w_dec: sae.decoder().w_dec.clone(),
        b_dec: sae.decoder().b_dec.clone(),
        d_in: d,
        d_sae: d,
    };
    sae = Sae::from_parts(
        cfg,
        enc,
        dec,
        Sparsifier::JumpReLU {
            thresholds: vec![0.5; d],
        },
    )
    .unwrap();
    let x = vec![-1.0f32, 0.5, 0.6, 1.0];
    let mut z = SparseOut::new(d as u32);
    sae.encode(&x, 1, &mut z).unwrap();
    // -1.0 → below threshold, drop. 0.5 → not strictly > 0.5, drop. 0.6, 1.0 → pass.
    assert_eq!(z.row_indices(0), &[2, 3]);
    assert!((z.row_values(0)[0] - 0.6).abs() < 1e-6);
    assert!((z.row_values(0)[1] - 1.0).abs() < 1e-6);
}

#[test]
fn relu_drops_nonpositive() {
    let d = 5;
    let mut sae = identity_sae(d, d);
    let cfg = sae.config().clone();
    let enc = EncoderWeights {
        w_enc: sae.encoder().w_enc.clone(),
        b_enc: sae.encoder().b_enc.clone(),
        d_in: d,
        d_sae: d,
    };
    let dec = DecoderWeights {
        w_dec: sae.decoder().w_dec.clone(),
        b_dec: sae.decoder().b_dec.clone(),
        d_in: d,
        d_sae: d,
    };
    sae = Sae::from_parts(cfg, enc, dec, Sparsifier::Relu).unwrap();
    let x = vec![-1.0f32, 0.0, 2.0, -0.5, 3.0];
    let mut z = SparseOut::new(d as u32);
    sae.encode(&x, 1, &mut z).unwrap();
    assert_eq!(z.row_indices(0), &[2, 4]);
}

#[test]
fn batched_encode_per_row_independence() {
    let d = 4;
    let sae = identity_sae(d, 2);
    let x = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]; // 2 batch rows
    let mut z = SparseOut::new(d as u32);
    sae.encode(&x, 2, &mut z).unwrap();
    assert_eq!(z.batch_size(), 2);
    assert_eq!(z.row_indices(0), &[2, 3]); // top-2 of [1,2,3,4]
    assert_eq!(z.row_indices(1), &[0, 1]); // top-2 of [4,3,2,1]
}

#[test]
fn shape_mismatch_returns_error() {
    let d = 4;
    let sae = identity_sae(d, 2);
    let x = vec![1.0; d - 1]; // wrong length
    let mut z = SparseOut::new(d as u32);
    assert!(sae.encode(&x, 1, &mut z).is_err());
}

// ---------- SAELens loader integration ----------

fn write_saelens_dir(dir: &std::path::Path) {
    let d_in = 4;
    let d_sae = 6;
    let k = 2;
    std::fs::create_dir_all(dir).unwrap();

    let cfg = serde_json::json!({
        "architecture": "topk",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "apply_b_dec_to_input": true,
        "normalize_activations": "none",
        "k": k,
    });
    std::fs::write(dir.join("cfg.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();

    // SAELens stores W_enc as [d_in, d_sae] in some checkpoints. We write it that
    // way and verify the loader transposes correctly.
    let w_enc: Vec<f32> = (0..d_in * d_sae).map(|i| (i as f32) * 0.01).collect();
    let w_dec: Vec<f32> = (0..d_sae * d_in).map(|i| (i as f32) * 0.02).collect();
    let b_enc: Vec<f32> = vec![0.1; d_sae];
    let b_dec: Vec<f32> = vec![0.2; d_in];

    let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
    tensors.insert(
        "W_enc".to_string(),
        TensorView::new(Dtype::F32, vec![d_in, d_sae], bytemuck::cast_slice(&w_enc)).unwrap(),
    );
    tensors.insert(
        "W_dec".to_string(),
        TensorView::new(Dtype::F32, vec![d_sae, d_in], bytemuck::cast_slice(&w_dec)).unwrap(),
    );
    tensors.insert(
        "b_enc".to_string(),
        TensorView::new(Dtype::F32, vec![d_sae], bytemuck::cast_slice(&b_enc)).unwrap(),
    );
    tensors.insert(
        "b_dec".to_string(),
        TensorView::new(Dtype::F32, vec![d_in], bytemuck::cast_slice(&b_dec)).unwrap(),
    );
    serialize_to_file(&tensors, &None, &dir.join("sae_weights.safetensors")).unwrap();
}

#[test]
fn loads_saelens_directory() {
    let tmp = tempfile::tempdir().unwrap();
    write_saelens_dir(tmp.path());
    let sae = Sae::load(tmp.path()).unwrap();
    assert_eq!(sae.d_in(), 4);
    assert_eq!(sae.d_sae(), 6);
    assert!(matches!(sae.sparsifier(), Sparsifier::TopK { k: 2, .. }));

    // Verify the loader transposed W_enc to internal [d_sae, d_in] layout.
    // Internal index [feature=0, d_in_idx=2] corresponds to the on-disk
    // W_enc[d_in_idx=2, feature=0], which is on-disk index 2*6 + 0 = 12 → 0.12.
    let expected = 12.0 * 0.01;
    let feature = 0usize;
    let d_in_idx = 2usize;
    let got = sae.encoder().w_enc[feature * 4 + d_in_idx];
    assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
}

#[test]
fn loader_round_trips_through_encode_decode() {
    let tmp = tempfile::tempdir().unwrap();
    write_saelens_dir(tmp.path());
    let sae = Sae::load(tmp.path()).unwrap();

    let x = vec![0.5f32, -0.5, 1.0, 0.25];
    let mut recon = vec![0.0f32; sae.d_in()];
    sae.reconstruct(&x, 1, &mut recon).unwrap();
    // We don't assert a specific value (the weights are arbitrary), only that
    // the pipeline runs without errors and produces finite output.
    for v in &recon {
        assert!(v.is_finite(), "non-finite reconstruction: {v}");
    }
}

// ---------- .sit format round-trip ----------

#[test]
fn sit_round_trip_topk() {
    let tmp = tempfile::tempdir().unwrap();
    write_saelens_dir(tmp.path());
    let sae = Sae::load(tmp.path()).unwrap();

    let sit_path = tmp.path().join("out.sit");
    sae.write_sit(&sit_path).unwrap();

    let loaded = Sae::load(&sit_path).unwrap();
    assert_eq!(loaded.d_in(), sae.d_in());
    assert_eq!(loaded.d_sae(), sae.d_sae());

    // Same encode on same input → same SparseFeatures.
    let x = vec![0.5f32, -0.5, 1.0, 0.25];
    let mut z1 = SparseOut::new(sae.d_sae() as u32);
    let mut z2 = SparseOut::new(loaded.d_sae() as u32);
    sae.encode(&x, 1, &mut z1).unwrap();
    loaded.encode(&x, 1, &mut z2).unwrap();
    assert_eq!(z1.indices, z2.indices);
    for (a, b) in z1.values.iter().zip(z2.values.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn sit_round_trip_jumprelu() {
    let d = 8;
    let sae = identity_sae(d, d);
    let cfg = sae.config().clone();
    let enc = saeitoshi::sae::EncoderWeights {
        w_enc: sae.encoder().w_enc.clone(),
        b_enc: sae.encoder().b_enc.clone(),
        d_in: d,
        d_sae: d,
    };
    let dec = saeitoshi::sae::DecoderWeights {
        w_dec: sae.decoder().w_dec.clone(),
        b_dec: sae.decoder().b_dec.clone(),
        d_in: d,
        d_sae: d,
    };
    let with_jumprelu = Sae::from_parts(
        cfg,
        enc,
        dec,
        Sparsifier::JumpReLU { thresholds: vec![0.5; d] },
    )
    .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let sit_path = tmp.path().join("jumprelu.sit");
    with_jumprelu.write_sit(&sit_path).unwrap();

    let loaded = Sae::load(&sit_path).unwrap();
    assert!(matches!(loaded.sparsifier(), Sparsifier::JumpReLU { .. }));

    let x = vec![-1.0f32, 0.5, 0.6, 1.0, 0.0, 2.0, -0.3, 0.51];
    let mut z = SparseOut::new(loaded.d_sae() as u32);
    loaded.encode(&x, 1, &mut z).unwrap();
    assert_eq!(z.row_indices(0), &[2, 3, 5, 7]);
}

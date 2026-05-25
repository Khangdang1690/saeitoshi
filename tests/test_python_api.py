"""End-to-end Python API tests. Exercise the PyO3 FFI boundary by loading
SAELens checkpoints from disk and running encode/decode/reconstruct.

These tests do not require sae_lens — they re-implement the SAELens math
in numpy so we can verify saeitoshi's output independently. The strict
≤1e-5 parity gate against the real sae_lens library lives in
``tests/test_parity_saelens.py`` (added when sae_lens is installable).
"""

from __future__ import annotations

import numpy as np
import pytest

import sae


def reference_encode_topk(
    x: np.ndarray,
    *,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    b_dec: np.ndarray,
    k: int,
    apply_b_dec_to_input: bool,
) -> np.ndarray:
    """SAELens topk encode in numpy. Returns dense [B, d_sae]."""
    x = x.copy()
    if apply_b_dec_to_input:
        x = x - b_dec
    pre = x @ w_enc + b_enc
    out = np.zeros_like(pre)
    for b in range(pre.shape[0]):
        idx = np.argsort(-pre[b], kind="stable")[:k]
        out[b, idx] = pre[b, idx]
    return out


def reference_encode_relu(
    x: np.ndarray,
    *,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    b_dec: np.ndarray,
    apply_b_dec_to_input: bool,
) -> np.ndarray:
    x = x.copy()
    if apply_b_dec_to_input:
        x = x - b_dec
    pre = x @ w_enc + b_enc
    return np.maximum(pre, 0.0)


def reference_encode_jumprelu(
    x: np.ndarray,
    *,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    b_dec: np.ndarray,
    threshold: np.ndarray,
    apply_b_dec_to_input: bool,
) -> np.ndarray:
    x = x.copy()
    if apply_b_dec_to_input:
        x = x - b_dec
    pre = x @ w_enc + b_enc
    mask = (pre > threshold).astype(np.float32)
    return mask * np.maximum(pre, 0.0)


def reference_decode(
    z: np.ndarray,
    *,
    w_dec: np.ndarray,
    b_dec: np.ndarray,
) -> np.ndarray:
    return z @ w_dec + b_dec


# ---------- import / load ----------

def test_sae_module_exposes_classes():
    assert hasattr(sae, "SAE")
    assert hasattr(sae, "SparseFeatures")
    assert hasattr(sae, "__version__")


def test_load_topk(make_sae):
    path, cfg, _ = make_sae("topk", d_in=8, d_sae=16, k=4)
    model = sae.SAE.load(str(path))
    assert model.d_in == 8
    assert model.d_sae == 16
    assert "topk" in model.architecture


def test_load_jumprelu(make_sae):
    path, _, _ = make_sae("jumprelu", d_in=8, d_sae=16, threshold=0.05)
    model = sae.SAE.load(str(path))
    assert "jumprelu" in model.architecture


def test_load_standard(make_sae):
    path, _, _ = make_sae("standard", d_in=8, d_sae=16)
    model = sae.SAE.load(str(path))
    assert "standard" in model.architecture


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises((IOError, ValueError)):
        sae.SAE.load(str(tmp_path / "does-not-exist"))


# ---------- encode / decode parity ----------

@pytest.mark.parametrize("apply_b_dec", [True, False])
def test_topk_encode_matches_numpy(make_sae, apply_b_dec):
    d_in, d_sae, k = 8, 16, 4
    path, _, tensors = make_sae(
        "topk", d_in=d_in, d_sae=d_sae, k=k, apply_b_dec_to_input=apply_b_dec
    )
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(42)
    x = rng.standard_normal((3, d_in), dtype=np.float32)

    features = model.encode(x)
    dense = features.to_dense()

    expected = reference_encode_topk(
        x,
        w_enc=tensors["W_enc"],
        b_enc=tensors["b_enc"],
        b_dec=tensors["b_dec"],
        k=k,
        apply_b_dec_to_input=apply_b_dec,
    )

    max_err = float(np.max(np.abs(dense - expected)))
    assert max_err < 1e-5, f"max abs error {max_err}"


def test_jumprelu_encode_matches_numpy(make_sae):
    d_in, d_sae = 8, 16
    path, _, tensors = make_sae("jumprelu", d_in=d_in, d_sae=d_sae, threshold=0.0)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, d_in), dtype=np.float32) * 0.5

    features = model.encode(x)
    dense = features.to_dense()

    expected = reference_encode_jumprelu(
        x,
        w_enc=tensors["W_enc"],
        b_enc=tensors["b_enc"],
        b_dec=tensors["b_dec"],
        threshold=tensors["threshold"],
        apply_b_dec_to_input=True,
    )

    max_err = float(np.max(np.abs(dense - expected)))
    assert max_err < 1e-5, f"max abs error {max_err}"


def test_standard_relu_encode_matches_numpy(make_sae):
    d_in, d_sae = 8, 16
    path, _, tensors = make_sae("standard", d_in=d_in, d_sae=d_sae)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, d_in), dtype=np.float32)

    features = model.encode(x)
    dense = features.to_dense()
    expected = reference_encode_relu(
        x,
        w_enc=tensors["W_enc"],
        b_enc=tensors["b_enc"],
        b_dec=tensors["b_dec"],
        apply_b_dec_to_input=True,
    )

    max_err = float(np.max(np.abs(dense - expected)))
    assert max_err < 1e-5, f"max abs error {max_err}"


def test_decode_matches_numpy(make_sae):
    d_in, d_sae, k = 8, 16, 4
    path, _, tensors = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(13)
    x = rng.standard_normal((3, d_in), dtype=np.float32)

    features = model.encode(x)
    dense_z = features.to_dense()
    recon = model.decode(features)

    expected = reference_decode(
        dense_z, w_dec=tensors["W_dec"], b_dec=tensors["b_dec"]
    )

    max_err = float(np.max(np.abs(recon - expected)))
    assert max_err < 1e-5, f"max abs error {max_err}"


def test_reconstruct_matches_encode_then_decode(make_sae):
    d_in, d_sae, k = 8, 16, 4
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(17)
    x = rng.standard_normal((3, d_in), dtype=np.float32)

    recon = model.reconstruct(x)
    features = model.encode(x)
    recon2 = model.decode(features)

    assert np.allclose(recon, recon2, atol=1e-6)


# ---------- SparseFeatures shape ----------

def test_sparse_features_shape(make_sae):
    d_in, d_sae, k = 8, 16, 3
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, d_in), dtype=np.float32)
    features = model.encode(x)

    assert features.batch_size == 5
    assert features.d_sae == d_sae
    assert len(features) == 5
    assert features.indices.dtype == np.int32
    assert features.values.dtype == np.float32
    # TopK: every batch row has exactly k entries.
    assert features.indices.shape == (5 * k,)
    assert features.values.shape == (5 * k,)
    assert features.row_offsets.shape == (6,)
    assert int(features.row_offsets[-1]) == 5 * k


def test_sparse_features_to_dense_shape(make_sae):
    d_in, d_sae, k = 8, 16, 3
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k)
    model = sae.SAE.load(str(path))

    x = np.zeros((4, d_in), dtype=np.float32)
    features = model.encode(x)
    dense = features.to_dense()
    assert dense.shape == (4, d_sae)
    assert dense.dtype == np.float32


# ---------- input validation ----------

def test_wrong_d_in_raises(make_sae):
    d_in = 8
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=16, k=4)
    model = sae.SAE.load(str(path))
    x = np.zeros((1, d_in + 1), dtype=np.float32)
    with pytest.raises(ValueError):
        model.encode(x)

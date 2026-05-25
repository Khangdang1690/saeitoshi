"""Tests for the streaming encode_stream/top_activations API and the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sae


# ---------- encode_stream ----------

def test_encode_stream_matches_one_shot(make_sae, tmp_path):
    d_in, d_sae, k = 8, 16, 3
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(0)
    n = 100
    x = rng.standard_normal((n, d_in), dtype=np.float32)

    # One-shot reference
    ref = model.encode(x).to_dense()

    # Streaming via in-memory ndarray
    pieces = []
    for batch in model.encode_stream(x, batch_size=16):
        pieces.append(batch.to_dense())
    streamed = np.concatenate(pieces, axis=0)

    assert ref.shape == streamed.shape
    assert np.allclose(ref, streamed, atol=1e-6)


def test_encode_stream_from_npy_file(make_sae, tmp_path):
    d_in, d_sae, k = 8, 16, 3
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(0)
    n = 50
    x = rng.standard_normal((n, d_in), dtype=np.float32)
    npy = tmp_path / "acts.npy"
    np.save(str(npy), x)

    pieces = []
    for batch in model.encode_stream(str(npy), batch_size=10):
        pieces.append(batch.to_dense())
    streamed = np.concatenate(pieces, axis=0)

    ref = model.encode(x).to_dense()
    assert np.allclose(ref, streamed, atol=1e-6)


def test_encode_stream_rejects_3d_input(make_sae):
    d_in, d_sae, k = 8, 16, 3
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k)
    model = sae.SAE.load(str(path))
    bad = np.zeros((2, 3, d_in), dtype=np.float32)
    with pytest.raises(ValueError):
        list(model.encode_stream(bad))


# ---------- top_activations ----------

def test_top_activations_finds_known_peaks(make_sae):
    d_in, d_sae, k = 8, 16, 3
    path, _, _ = make_sae("topk", d_in=d_in, d_sae=d_sae, k=k, seed=7)
    model = sae.SAE.load(str(path))

    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal((n, d_in), dtype=np.float32)

    # Compute the "ground truth" max-activating row per feature in one pass.
    dense = model.encode(x).to_dense()  # [n, d_sae]
    feature_ids = [3, 7, 11]

    result = model.top_activations(x, feature_ids=feature_ids, top_k=5, batch_size=32)

    for fid in feature_ids:
        # Reference: rows with non-zero activation for this feature,
        # sorted descending by value.
        col = dense[:, fid]
        nonzero_rows = np.flatnonzero(col)
        ref = sorted(((float(col[r]), int(r)) for r in nonzero_rows), reverse=True)[:5]

        got = result[fid]
        # Both lists are tuples (value, token_idx); compare values to 1e-5.
        assert len(got) == len(ref)
        for (gv, gi), (rv, ri) in zip(got, ref):
            assert abs(gv - rv) < 1e-5
            # Index equality required when no value ties.
            assert gi == ri


def test_top_activations_empty_feature_list(make_sae):
    path, _, _ = make_sae("topk", d_in=4, d_sae=8, k=2)
    model = sae.SAE.load(str(path))
    x = np.zeros((10, 4), dtype=np.float32)
    assert model.top_activations(x, feature_ids=[]) == {}


# ---------- CLI ----------

def _run_cli(*args, expect_zero=True) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "sae.cli", *args],
        capture_output=True,
        text=True,
    )
    if expect_zero:
        assert proc.returncode == 0, f"stderr: {proc.stderr}"
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_inspect_dumps_metadata(make_sae):
    path, _, _ = make_sae("topk", d_in=8, d_sae=16, k=4)
    _, stdout, _ = _run_cli("inspect", str(path))
    data = json.loads(stdout)
    assert data["d_in"] == 8
    assert data["d_sae"] == 16
    assert "topk" in data["architecture"]


def test_cli_bench_prints_throughput(make_sae):
    path, _, _ = make_sae("topk", d_in=8, d_sae=16, k=4)
    _, stdout, _ = _run_cli(
        "bench", "--sae", str(path), "--tokens", "1000", "--batch-size", "100"
    )
    assert "saeitoshi:" in stdout
    assert "tokens/sec" in stdout

"""Shared fixtures for the saeitoshi test suite.

Builds tiny SAELens-format SAEs on disk for parity and round-trip tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


def write_saelens_checkpoint(
    path: Path,
    *,
    architecture: str,
    d_in: int,
    d_sae: int,
    k: int | None = None,
    threshold: float | np.ndarray | None = None,
    apply_b_dec_to_input: bool = True,
    normalize_activations: str = "none",
    seed: int = 0,
) -> dict:
    """Materialize a SAELens-format checkpoint at `path`.

    Returns the cfg dict and the in-memory weight tensors so tests can
    re-implement the math for parity checks without a full SAELens dep.
    """
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)

    # SAELens-style shapes: W_enc is [d_in, d_sae], W_dec is [d_sae, d_in].
    w_enc = rng.standard_normal((d_in, d_sae), dtype=np.float32) * 0.1
    w_dec = rng.standard_normal((d_sae, d_in), dtype=np.float32) * 0.1
    b_enc = rng.standard_normal((d_sae,), dtype=np.float32) * 0.01
    b_dec = rng.standard_normal((d_in,), dtype=np.float32) * 0.01

    tensors: dict[str, np.ndarray] = {
        "W_enc": w_enc,
        "W_dec": w_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
    }

    cfg: dict = {
        "architecture": architecture,
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "apply_b_dec_to_input": apply_b_dec_to_input,
        "normalize_activations": normalize_activations,
    }
    if architecture == "topk":
        assert k is not None
        cfg["k"] = k
    if architecture in ("jumprelu", "batch_topk"):
        if threshold is None:
            threshold_arr = np.full((d_sae,), 0.0, dtype=np.float32)
        elif np.isscalar(threshold):
            threshold_arr = np.full((d_sae,), float(threshold), dtype=np.float32)
        else:
            threshold_arr = np.asarray(threshold, dtype=np.float32)
        tensors["threshold"] = threshold_arr

    save_file(tensors, str(path / "sae_weights.safetensors"))
    (path / "cfg.json").write_text(json.dumps(cfg, indent=2))

    return {"cfg": cfg, "tensors": tensors}


@pytest.fixture
def make_sae(tmp_path):
    """Factory fixture: returns a callable that writes a SAELens checkpoint
    into a fresh subdirectory of `tmp_path` and returns (path, cfg, tensors).
    """
    counter = {"n": 0}

    def _make(architecture: str, **kwargs) -> tuple[Path, dict, dict]:
        counter["n"] += 1
        sub = tmp_path / f"sae_{counter['n']}"
        info = write_saelens_checkpoint(sub, architecture=architecture, **kwargs)
        return sub, info["cfg"], info["tensors"]

    return _make

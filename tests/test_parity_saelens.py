"""Strict parity test against the SAELens reference library.

Generated tiny SAEs are written to SAELens-format directories; both
sae_lens and saeitoshi load and encode them on the same input. Max
absolute error must be ≤ 1e-5 — the launch-blocking parity gate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import sae

sae_lens = pytest.importorskip("sae_lens")
torch = pytest.importorskip("torch")


def _build_topk_saelens_dir(tmp_path, d_in: int, d_sae: int, k: int, seed: int):
    """Build a SAELens-compatible TopK SAE directory.

    SAELens >=6 stores W_enc shape [d_in, d_sae] and W_dec shape [d_sae, d_in]
    in `sae_weights.safetensors`. cfg.json includes the v6 metadata shape.
    """
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    w_enc = rng.standard_normal((d_in, d_sae), dtype=np.float32) * 0.1
    w_dec = rng.standard_normal((d_sae, d_in), dtype=np.float32) * 0.1
    b_enc = rng.standard_normal((d_sae,), dtype=np.float32) * 0.01
    b_dec = rng.standard_normal((d_in,), dtype=np.float32) * 0.01

    save_file(
        {
            "W_enc": w_enc,
            "W_dec": w_dec,
            "b_enc": b_enc,
            "b_dec": b_dec,
        },
        str(tmp_path / "sae_weights.safetensors"),
    )
    cfg = {
        "architecture": "topk",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "k": k,
        "model_class_name": "HookedTransformer",
        "metadata": {
            "model_name": "test",
            "hook_name": "test.hook",
            "hook_head_index": None,
            "sae_lens_version": "6.44.0",
            "activation_fn": "topk",
        },
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))


@pytest.mark.parametrize("apply_b_dec", [True, False])
def test_topk_parity_vs_saelens(tmp_path, apply_b_dec):
    d_in, d_sae, k = 16, 64, 4
    _build_topk_saelens_dir(tmp_path, d_in, d_sae, k, seed=11)
    if not apply_b_dec:
        cfg_path = tmp_path / "cfg.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["apply_b_dec_to_input"] = False
        cfg_path.write_text(json.dumps(cfg))

    our = sae.SAE.load(str(tmp_path))

    try:
        sl = sae_lens.SAE.load_from_disk(str(tmp_path), device="cpu")
    except (AttributeError, TypeError):
        # API drift across sae_lens versions: try from_pretrained instead.
        sl = sae_lens.SAE.from_pretrained(str(tmp_path), device="cpu")[0]
    sl.eval()

    rng = np.random.default_rng(99)
    x = rng.standard_normal((5, d_in), dtype=np.float32)

    with torch.no_grad():
        sl_out = sl.encode(torch.from_numpy(x)).cpu().numpy()
    our_out = our.encode(x).to_dense()

    max_err = float(np.max(np.abs(sl_out - our_out)))
    assert max_err <= 1e-5, (
        f"saeitoshi vs sae_lens TopK max abs error {max_err} (>1e-5). "
        f"apply_b_dec_to_input={apply_b_dec}"
    )

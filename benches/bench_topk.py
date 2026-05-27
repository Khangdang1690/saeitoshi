"""TopK-focused micro-bench. Measures end-to-end `encode` time across a
sweep of (d_in, d_sae, batch) shapes — the TopK kernel is the dominant
cost at the v0.2+ headline shapes, so this isolates regressions in the
heap kernel from regressions in the matmul.

Usage (with the project's .venv active):
    python benches/bench_topk.py
    python benches/bench_topk.py --shapes 2048,16384,4096 --k 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


@dataclass(frozen=True)
class Shape:
    d_in: int
    d_sae: int
    batch: int

    @property
    def label(self) -> str:
        return f"d_in={self.d_in:>5} d_sae={self.d_sae:>6} B={self.batch:>5}"


DEFAULT_SHAPES = [
    Shape(d_in=2048, d_sae=16384, batch=512),
    Shape(d_in=2048, d_sae=16384, batch=4096),
    Shape(d_in=768, d_sae=12288, batch=4096),
    Shape(d_in=512, d_sae=8192, batch=4096),
]


def write_saelens_dir(dir_path: Path, shape: Shape, k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    w_enc = rng.standard_normal((shape.d_in, shape.d_sae), dtype=np.float32) * 0.1
    w_dec = rng.standard_normal((shape.d_sae, shape.d_in), dtype=np.float32) * 0.1
    b_enc = rng.standard_normal((shape.d_sae,), dtype=np.float32) * 0.01
    b_dec = rng.standard_normal((shape.d_in,), dtype=np.float32) * 0.01
    save_file(
        {"W_enc": w_enc, "W_dec": w_dec, "b_enc": b_enc, "b_dec": b_dec},
        str(dir_path / "sae_weights.safetensors"),
    )
    cfg = {
        "architecture": "topk",
        "d_in": shape.d_in,
        "d_sae": shape.d_sae,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "k": k,
        "model_class_name": "HookedTransformer",
        "metadata": {
            "model_name": "bench",
            "hook_name": "bench.hook",
            "hook_head_index": None,
            "sae_lens_version": "6.44.0",
            "activation_fn": "topk",
        },
    }
    (dir_path / "cfg.json").write_text(json.dumps(cfg))


def time_call(fn, warm: int, iters: int) -> float:
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        action="append",
        default=None,
        type=lambda s: Shape(*[int(x) for x in s.split(",")]),
    )
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--iters", type=int, default=11)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    shapes = args.shapes or DEFAULT_SHAPES

    import sae as saeitoshi

    print(
        f"# bench_topk — heap TopK, k={args.k}, warm={args.warm}, "
        f"iters={args.iters} (median), cpu_count={os.cpu_count()}"
    )
    print()
    print(f"| {'shape':<32} | {'encode (ms)':>12} |")
    print(f"|{'-' * 34}|{'-' * 14}|")

    for shape in shapes:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            write_saelens_dir(td_path, shape, k=args.k, seed=args.seed)
            si_sae = saeitoshi.SAE.load(str(td_path))
            rng = np.random.default_rng(shape.batch * 31 + args.seed)
            x = rng.standard_normal((shape.batch, shape.d_in), dtype=np.float32).astype(
                np.float32, copy=False
            )
            t = time_call(lambda: si_sae.encode(x), warm=args.warm, iters=args.iters)
        print(f"| {shape.label:<32} | {t * 1000:>11.1f}  |")


if __name__ == "__main__":
    main()

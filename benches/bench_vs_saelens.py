"""Side-by-side encode benchmark: sae_lens 6.44 vs saeitoshi.

Runs both libraries on the same TopK SAE shapes, prints throughput and
the saeitoshi/sae_lens ratio. Not part of the CI test gate — invoke
locally after kernel changes to validate the headline numbers.

Usage (with the project's .venv active):
    python benches/bench_vs_saelens.py
    python benches/bench_vs_saelens.py --shapes 2048,16384,512
    python benches/bench_vs_saelens.py --warm 5 --iters 30

The default shape grid matches README.md so the printed table can be
pasted in directly.
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
import torch
from safetensors.numpy import save_file

import sae as saeitoshi
import sae_lens


@dataclass(frozen=True)
class Shape:
    d_in: int
    d_sae: int
    batch: int

    @property
    def label(self) -> str:
        return f"d_in={self.d_in:>5} d_sae={self.d_sae:>6} B={self.batch:>5}"


DEFAULT_SHAPES = [
    Shape(d_in=512, d_sae=8192, batch=4096),
    Shape(d_in=768, d_sae=12288, batch=4096),
    Shape(d_in=2048, d_sae=16384, batch=4096),
    Shape(d_in=2048, d_sae=16384, batch=512),
]


def write_saelens_dir(dir_path: Path, shape: Shape, k: int, seed: int) -> None:
    """Generate a SAELens-format SAE on disk so both libraries can load the
    same weights (same approach as `tests/test_parity_saelens.py`)."""
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
    """Median wall-clock seconds across `iters` runs after `warm` warmups."""
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


def parse_shape(s: str) -> Shape:
    d_in, d_sae, batch = (int(x) for x in s.split(","))
    return Shape(d_in=d_in, d_sae=d_sae, batch=batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        action="append",
        default=None,
        type=parse_shape,
        help="d_in,d_sae,batch (repeatable). Defaults to the README grid.",
    )
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--iters", type=int, default=11)
    parser.add_argument("--k", type=int, default=32, help="TopK k value")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--no-saelens",
        action="store_true",
        help="Skip sae_lens timing (useful when iterating on saeitoshi alone).",
    )
    args = parser.parse_args()
    shapes = args.shapes or DEFAULT_SHAPES

    backend = os.environ.get("SAEITOSHI_BACKEND", "tiled (default)")
    print(
        f"# bench_vs_saelens — sae_lens {sae_lens.__version__} vs saeitoshi "
        f"{saeitoshi.__version__}"
    )
    print(
        f"# TopK k={args.k}, warm={args.warm}, iters={args.iters} (median), "
        f"saeitoshi backend={backend}, cpu_count={os.cpu_count()}"
    )
    print()
    print(
        f"| {'shape':<32} | {'saeitoshi (ms)':>14} | {'sae_lens (ms)':>14} | "
        f"{'speedup':>9} |"
    )
    print(f"|{'-' * 34}|{'-' * 16}|{'-' * 16}|{'-' * 11}|")

    for shape in shapes:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            write_saelens_dir(td_path, shape, k=args.k, seed=args.seed)

            si_sae = saeitoshi.SAE.load(str(td_path))
            try:
                sl_sae = sae_lens.SAE.load_from_disk(str(td_path), device="cpu")
            except (AttributeError, TypeError):
                sl_sae = sae_lens.SAE.from_pretrained(str(td_path), device="cpu")[0]
            sl_sae.eval()

            rng = np.random.default_rng(shape.batch * 31 + args.seed)
            x = rng.standard_normal(
                (shape.batch, shape.d_in), dtype=np.float32
            ).astype(np.float32, copy=False)

            si_time = time_call(
                lambda: si_sae.encode(x), warm=args.warm, iters=args.iters
            )

            if args.no_saelens:
                sl_time = float("nan")
                speedup = "n/a"
            else:
                x_torch = torch.from_numpy(x)
                with torch.no_grad():
                    sl_time = time_call(
                        lambda: sl_sae.encode(x_torch),
                        warm=args.warm,
                        iters=args.iters,
                    )
                speedup = f"{sl_time / si_time:>7.2f}x"

            print(
                f"| {shape.label:<32} | {si_time * 1000:>13.1f}  | "
                f"{sl_time * 1000:>13.1f}  | {speedup:>9} |"
            )


if __name__ == "__main__":
    main()

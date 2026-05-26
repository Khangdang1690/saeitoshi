"""TopK-focused micro-bench. Compares v0.2 heap path against the legacy
full-sort by toggling `SAEITOSHI_TOPK=legacy` between runs.

The matmul portion of encode is identical across runs (same perf-v2
backend), so the per-shape delta isolates the TopK improvement.

Usage (with the project's .venv active):
    python benches/bench_topk.py
    python benches/bench_topk.py --shapes 2048,16384,4096 --k 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
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


def run_for_backend(shapes, k, seed, warm, iters):
    # Import lazily so SAEITOSHI_TOPK env is observed at first call.
    import sae as saeitoshi

    rows = []
    for shape in shapes:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            write_saelens_dir(td_path, shape, k=k, seed=seed)
            si_sae = saeitoshi.SAE.load(str(td_path))
            rng = np.random.default_rng(shape.batch * 31 + seed)
            x = rng.standard_normal(
                (shape.batch, shape.d_in), dtype=np.float32
            ).astype(np.float32, copy=False)
            t = time_call(lambda: si_sae.encode(x), warm=warm, iters=iters)
            rows.append((shape, t))
    return rows


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
    parser.add_argument(
        "--mode",
        choices=("compare", "heap", "legacy"),
        default="compare",
        help=(
            "compare: run both legacy and heap in child processes "
            "(so SAEITOSHI_TOPK is honored). heap/legacy: run once "
            "in the current process honoring the current env."
        ),
    )
    args = parser.parse_args()
    shapes = args.shapes or DEFAULT_SHAPES

    if args.mode in ("heap", "legacy"):
        os.environ["SAEITOSHI_TOPK"] = args.mode if args.mode == "legacy" else ""
        rows = run_for_backend(shapes, args.k, args.seed, args.warm, args.iters)
        for shape, t in rows:
            print(f"{shape.label}  {t * 1000:>9.2f} ms")
        return

    # `compare` mode: spawn one child per backend, capture median timings,
    # diff them.
    print(
        f"# bench_topk — TopK isolated comparison, k={args.k}, "
        f"warm={args.warm}, iters={args.iters} (median), "
        f"cpu_count={os.cpu_count()}"
    )
    print()
    print(
        f"| {'shape':<32} | {'legacy (ms)':>12} | {'heap (ms)':>12} | "
        f"{'speedup':>9} |"
    )
    print(f"|{'-' * 34}|{'-' * 14}|{'-' * 14}|{'-' * 11}|")

    shape_args: list[str] = []
    for shape in shapes:
        shape_args += ["--shapes", f"{shape.d_in},{shape.d_sae},{shape.batch}"]

    def child(mode: str) -> dict[str, float]:
        env = dict(os.environ)
        if mode == "legacy":
            env["SAEITOSHI_TOPK"] = "legacy"
        else:
            env.pop("SAEITOSHI_TOPK", None)
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--mode",
                mode,
                "--k",
                str(args.k),
                "--seed",
                str(args.seed),
                "--warm",
                str(args.warm),
                "--iters",
                str(args.iters),
                *shape_args,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        out = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # "d_in=  512 d_sae=  8192 B= 4096   1234.56 ms"
            label, _, rest = line.rpartition("  ")
            t_ms = float(rest.split()[0])
            out[label.strip()] = t_ms
        return out

    legacy = child("legacy")
    heap = child("heap")

    for shape in shapes:
        l = legacy.get(shape.label, float("nan"))
        h = heap.get(shape.label, float("nan"))
        speedup = f"{l / h:>7.2f}x" if h > 0 else "n/a"
        print(
            f"| {shape.label:<32} | {l:>11.1f}  | {h:>11.1f}  | {speedup:>9} |"
        )


if __name__ == "__main__":
    main()

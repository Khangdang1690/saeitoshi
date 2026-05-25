"""Command-line interface for saeitoshi.

Registered as ``saeitoshi`` in ``pyproject.toml`` ``[project.scripts]``.

Subcommands:
    inspect PATH                      Dump SAE metadata.
    bench   --sae PATH [--tokens N]   Throughput benchmark vs an in-process
                                      numpy reference (and SAELens when
                                      installed and ``--saelens`` is passed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import sae


def _bytes(n: int) -> str:
    for unit, suffix in [(1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")]:
        if n >= unit:
            return f"{n / unit:.2f} {suffix}"
    return f"{n} B"


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path)
    model = sae.SAE.load(str(path))
    info = {
        "path": str(path),
        "architecture": model.architecture,
        "d_in": model.d_in,
        "d_sae": model.d_sae,
        "expansion_ratio": round(model.d_sae / model.d_in, 2),
        "weight_bytes_estimate": _bytes(2 * model.d_in * model.d_sae * 4),
    }
    print(json.dumps(info, indent=2))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    model = sae.SAE.load(str(args.sae))
    d_in = model.d_in
    d_sae = model.d_sae
    n_tokens = args.tokens
    batch_size = args.batch_size

    rng = np.random.default_rng(args.seed)
    print(
        f"benching {model.architecture} SAE: d_in={d_in}, d_sae={d_sae}, "
        f"tokens={n_tokens}, batch={batch_size}"
    )

    x = rng.standard_normal((n_tokens, d_in), dtype=np.float32)

    # Warm up to fault in pages.
    _ = model.encode(x[:batch_size])

    t0 = time.perf_counter()
    total = 0
    for start in range(0, n_tokens, batch_size):
        end = min(start + batch_size, n_tokens)
        _ = model.encode(np.ascontiguousarray(x[start:end]))
        total += end - start
    saei_t = time.perf_counter() - t0
    print(
        f"  saeitoshi: {saei_t * 1000:.1f} ms total, "
        f"{total / saei_t:,.0f} tokens/sec"
    )

    if args.saelens:
        try:
            import sae_lens  # noqa: F401
            import torch
        except ImportError:
            print("  --saelens passed but sae_lens / torch not installed", file=sys.stderr)
            return 1
        # Load SAELens from the same directory (must be a SAELens-format SAE).
        sl_model = sae_lens.SAE.load_from_pretrained(str(args.sae)).eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            for start in range(0, n_tokens, batch_size):
                end = min(start + batch_size, n_tokens)
                tb = torch.from_numpy(x[start:end])
                _ = sl_model.encode(tb)
            sl_t = time.perf_counter() - t0
        print(
            f"  sae_lens:  {sl_t * 1000:.1f} ms total, "
            f"{total / sl_t:,.0f} tokens/sec"
        )
        print(f"  speedup:   {sl_t / saei_t:.2f}x")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saeitoshi", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Dump SAE metadata as JSON.")
    p_inspect.add_argument("path", help="SAE directory or .sit file.")
    p_inspect.set_defaults(func=cmd_inspect)

    p_bench = sub.add_parser("bench", help="Throughput benchmark.")
    p_bench.add_argument("--sae", required=True, help="SAE directory or .sit file.")
    p_bench.add_argument("--tokens", type=int, default=100_000, help="Total tokens.")
    p_bench.add_argument("--batch-size", type=int, default=1024)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument(
        "--saelens",
        action="store_true",
        help="Also benchmark sae_lens (must be installed).",
    )
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

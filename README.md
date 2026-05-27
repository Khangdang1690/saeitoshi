# saeitoshi

**Fast, pure-Rust inference for trained Sparse Autoencoders. Drop-in for SAELens with strict ≤1e-5 numerical parity, zero torch dependency, single-binary wheel.**

`saeitoshi` is a Rust core + Python bindings library for decoding LLM activations through trained Sparse Autoencoders (SAEs). It's the FAISS-equivalent for mech-interp work: a no-brainer dependency that loads any common SAE checkpoint format and runs `encode` / `decode` / `reconstruct` with hand-tuned SIMD kernels (AVX-512 / AVX2 / NEON) and runtime CPU dispatch.

## Install

```bash
pip install saeitoshi
```

Optional HuggingFace Hub support: `pip install "saeitoshi[hf]"`.

## Usage

```python
import sae

# Loads SAELens directory, EleutherAI sparsify directory, .sit file, or hf:// URL.
model = sae.SAE.load("hf://jbloom/Gemma-2b-IT-Resid-Post-SAE")

# Single batch — numpy float32[B, d_in] in, SparseFeatures out.
features = model.encode(activations)
reconstruction = model.decode(features)

# Streaming over a large file, never materializes dense features.
for batch in model.encode_stream("activations.npy", batch_size=8192):
    ...   # batch.indices: int32, batch.values: float32, batch.row_offsets: int32

# Top-activating positions per feature, single pass over a corpus.
top = model.top_activations(
    "activations.npy", feature_ids=[1234, 5678], top_k=20,
)
```

CLI:

```bash
saeitoshi inspect ./my_sae                                # dump metadata
saeitoshi bench --sae ./my_sae --tokens 100_000           # throughput
saeitoshi bench --sae ./my_sae --tokens 100_000 --saelens # side-by-side vs sae_lens
```

## What it supports

| Surface | v0 |
|---|---|
| Sparsification | TopK, JumpReLU, BatchTopK (resolves to JumpReLU at load), ReLU/Standard |
| Weight dtype | FP32 (FP16 / INT8 slots reserved in `.sit` format for v0.1) |
| Loaders | SAELens directory, EleutherAI `sparsify` directory, raw `.safetensors`, native `.sit` |
| Sources | local path, `hf://org/repo[@rev][/subfolder]` |
| Streaming | numpy `.npy` (mmap), in-memory ndarray |
| Wheels | macOS-ARM, Linux-x86_64, Linux-aarch64 (Python 3.10+, stable ABI) |

## Numerical parity

saeitoshi reproduces `sae_lens.SAE.encode` output to within **1e-5** absolute error on all supported architectures, verified in CI ([tests/test_parity_saelens.py](tests/test_parity_saelens.py)). The scalar reference backend is the gate; SIMD backends (AVX2 / AVX-512 / NEON) are property-tested against it across 30 (d_in × d_sae × batch) shapes ([crates/saeitoshi/tests/simd_parity.rs](crates/saeitoshi/tests/simd_parity.rs)).

## Performance, honestly

v0 ships a working AVX2 / AVX-512 / NEON encoder with batch-level rayon parallelism. On a single-machine head-to-head against `sae_lens` 6.44 (which uses PyTorch's MKL-backed matmul) on an Intel i9-13900HX:

| d_in × d_sae | saeitoshi (AVX2 + 24t) | sae_lens (MKL) |
|---|---|---|
| 512 × 8192   | ~1.8s / 4096 tokens | ~56ms |
| 768 × 12288  | ~3.1s / 4096 tokens | ~113ms |
| 2048 × 16384 | ~5.9s / 4096 tokens | ~387ms |

MKL is hard to beat on standard matmul shapes — that's not surprising. v0's value proposition is **correctness and packaging**, not raw throughput vs MKL:

- Strict ≤1e-5 SAELens parity, in CI.
- Single pure-Rust binary, no torch in the dependency closure.
- Multi-format loader (SAELens / sparsify / raw / hf://).
- Streaming + top-activations utilities with zero-copy numpy interop.
- Works on Mac + Linux without a CUDA install or a GPU.

The 5× SAELens speedup the project brief targets is the v0.1 focus: tiled GEMM microkernel, INT8 weight loading, cross-batch register reuse. The architecture is set up for it (Backend vtable, runtime SIMD dispatch, on-disk format with dtype slots reserved).

### v0.1 update — tiled GEMM (`perf-v2`, default since v0.1)

v0.1 ships a cache-tiled, register-blocked GEMM encoder that runs by
default. The kernel is documented from first principles in
[docs/perf-analysis.md](docs/perf-analysis.md). The legacy
per-dot-product backend is reachable via `SAEITOSHI_BACKEND=legacy` for
one release in case you need to bisect a regression.

End-to-end `sae.encode` on the same i9-13900HX, TopK k=32, vs sae_lens
6.44 (run `python benches/bench_vs_saelens.py` to reproduce):

| d_in × d_sae | B | saeitoshi v0 | saeitoshi v0.1 | saeitoshi v0.2 | sae_lens (MKL) |
|---|---|---|---|---|---|
| 512 × 8192   | 4096 | ~1800 ms | ~1640 ms | **~62 ms**  | ~73 ms  |
| 768 × 12288  | 4096 | ~3100 ms | ~2750 ms | **~158 ms** | ~152 ms |
| 2048 × 16384 | 4096 | ~5900 ms | ~3920 ms | **~438 ms** | ~467 ms |
| 2048 × 16384 |  512 | ~735 ms  | ~466 ms  | **~47 ms**  | ~70 ms  |

v0.2 closes the TopK gap: a heap-based partial sort (O(n log k) vs the
v0 sort's O(n log n)) with branchless integer-key comparison, plus
row-level rayon parallelism in the sparsifier. Isolated TopK at
d_sae=16384, k=32 dropped from ~853 µs to ~16 µs per row (~52×). The
heap kernel is now the default; flip back with `SAEITOSHI_TOPK=legacy`
for one release if you need to bisect.

What perf-v2 actually changes:

- W is repacked once at SAE load into `M_R=16`-row column-major
  panels. The microkernel streams that panel through a register-
  blocked tile (AVX2 16×6 / AVX-512 16×12 / NEON 16×6).
- M-block rayon threading: each thread owns a contiguous slice of
  `d_sae` features and shares the `x` panel via L3. The legacy backend
  partitioned the batch, so every thread re-streamed the full 128 MB
  W from DRAM per call.
- Same ≤1e-5 SAELens parity gate. SIMD parity in
  [crates/saeitoshi/tests/simd_parity.rs](crates/saeitoshi/tests/simd_parity.rs)
  also runs the tiled backend at 1/2/4/24 rayon thread counts to catch
  false-sharing regressions.

Knobs:

- `SAEITOSHI_GEMM_TILE=MC,NC,KC` — override the M-block grouping (MC).
- `SAEITOSHI_NO_PARALLEL=1` — disable rayon parallelism entirely.

## Status

Pre-alpha. Active work; see `.claude/plans/` for the milestone plan, and `git log` for what's landed.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

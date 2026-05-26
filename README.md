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

## Status

Pre-alpha. Active work; see `.claude/plans/` for the milestone plan, and `git log` for what's landed.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

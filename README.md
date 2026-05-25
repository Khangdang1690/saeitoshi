# saeitoshi

> _Decoding 1B token activations through a 4M-feature SAE takes SAELens ~6 days. saeitoshi does it in [TBD] — on a laptop._

<!-- benchmark numbers replace the TBD once M3 lands. -->

**saeitoshi** is a fast inference engine for trained Sparse Autoencoders (SAEs) — the tools mechanistic-interpretability researchers use to decompose LLM activations into human-interpretable features. It's the FAISS-equivalent for SAEs: a Rust core with hand-tuned SIMD kernels (AVX-512 / AVX2 / NEON) and Python bindings, designed to drop in for [SAELens](https://github.com/jbloomAus/SAELens) on the inference path while running ≥5× faster on CPU.

## Install

```bash
pip install saeitoshi
```

Optional extras: `pip install "saeitoshi[hf]"` to load SAEs directly from HuggingFace Hub.

## Usage

```python
import sae

# Load any common format — SAELens directory, EleutherAI sparsify, raw safetensors, or .sit
model = sae.SAE.load("hf://jbloom/Gemma-2b-IT-Resid-Post-SAE")

# Encode a batch of activations — numpy contiguous float32, shape [B, d_in]
features = model.encode(activations)               # -> SparseFeatures
reconstruction = model.decode(features)            # -> numpy float32[B, d_in]

# Streaming over a large file, never materializes dense features
for batch in model.encode_stream("activations.npy", batch_size=8192):
    ...   # batch.indices: int32[B, k], batch.values: float32[B, k]
```

## What it supports

| Surface | v0 |
|---|---|
| Sparsification | TopK, JumpReLU, BatchTopK, ReLU, Gated |
| Weight dtype | FP32, FP16, INT8 (loaded, not quantized at runtime) |
| Loaders | SAELens, EleutherAI `sparsify`, raw safetensors, `.sit` |
| Sources | `hf://repo_id`, local dir, local file |
| Streaming | numpy `.npy` (mmap), safetensors |
| Wheels | macOS-ARM, Linux-x86_64, Linux-aarch64 (Python 3.10+) |

## Numerical parity

saeitoshi reproduces SAELens encoder output to within **1e-5** absolute error on FP32, validated in CI against every supported architecture × dtype combination. See [tests/test_parity.py](tests/test_parity.py).

## Status

Pre-alpha. Active work in progress; see the milestone plan in `.claude/plans/` for the build order.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

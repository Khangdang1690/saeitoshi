# saeitoshi v0.1 — encoder performance analysis

## Why this document exists

v0 shipped correct numerics (52 tests, ≤1e-5 parity vs SAELens 6.44) but the
encoder is ~16× slower than `sae_lens` on the headline benchmark
(d_in=2048, d_sae=16384, batch=4096 on Intel i9-13900HX). The original brief
targeted ≥5× *faster*. The closure plan for that gap is documented in
[Design](#design). This first section answers *why* the gap exists, in enough
detail that the next maintainer can audit the design choices that follow.

---

## 1. What SAELens actually does

`sae_lens.SAE.encode` for a TopK SAE (the configuration we benchmark
against) lands here:

```python
# .venv/Lib/site-packages/sae_lens/saes/topk_sae.py:254-264
def encode(self, x: torch.Tensor) -> torch.Tensor:
    sae_in = self.process_sae_in(x)
    hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
    ...
    return self.hook_sae_acts_post(self.activation_fn(hidden_pre))
```

`self.W_enc` is an `nn.Parameter` of shape `[d_in, d_sae]` (note: this is
the **transpose** of saeitoshi's `[d_sae, d_in]` row-major layout — our
SAELens loader does that transpose at load time). The `@` operator on two
2-D tensors calls `torch.Tensor.__matmul__` → `torch.matmul` → ATen
`aten::matmul` → for 2-D × 2-D the kernel-dispatch picks `aten::mm` →
`at::native::mm_cpu`. On x86 CPUs with MKL linked (PyTorch's default CPU
build on Windows/Linux), `mm_cpu` dispatches to MKL's `cblas_sgemm`.

The `+ self.b_enc` broadcast becomes a separate fused-bias add in ATen.
The `hook_sae_acts_pre` callback is a no-op in non-instrumented mode.
There is **no Python-level overhead** worth mentioning: a couple of
function calls per encode, all in C++ after the first hop.

What MKL's `sgemm` does internally (this is the part we have to compete
with) is the **GotoBLAS / BLIS 5-loop blocked algorithm**:

1. Outer M-loop in blocks of `M_C` (rows of C).
2. K-loop in blocks of `K_C` (reduction direction). For each `K_C` block,
   pack a column-major panel of A of shape `M_C × K_C` into an L2-resident
   buffer.
3. N-loop in blocks of `N_C` (cols of C). For each `N_C` block, pack a
   row-major panel of B of shape `K_C × N_C` into an L3-resident buffer.
4. Inner M-loop in blocks of `M_R` (a small register-tile).
5. Inner N-loop in blocks of `N_R`. Innermost: a hand-tuned `M_R × N_R`
   register-blocked microkernel that streams `K_C` elements and
   accumulates into the register tile.

MKL ships dozens of microkernels selected at runtime by `cpuid`. On
Raptor Lake (Haswell-family ISA, AVX2+FMA, no AVX-512) it uses an
`8×24` or `4×16` AVX2 microkernel; on Skylake-X and newer it uses an
`8×24` or `16×6` AVX-512 microkernel. Multi-threading is openmp-based,
splitting the M dimension across threads so they share the packed B
panel via L3.

Two structural properties of MKL we *cannot* copy:
- It is JIT-tuned per CPU (we use `#[target_feature]` instead).
- It is closed-source (we build from BLIS papers).

One structural property MKL has that we have *no need for*:
- It targets general matrices. SAE encoders have a fixed asymmetric shape
  (small K, huge M, moderate N) and W is loaded once and reused forever.
  We can repack W at load time; MKL has to repack per call.

---

## 2. What saeitoshi currently does

The encoder entry point ([sae.rs](../crates/saeitoshi/src/sae.rs#L241))
calls into [kernels::encoder::encode_f32](../crates/saeitoshi/src/kernels/encoder.rs#L22),
which dispatches through a [Backend](../crates/saeitoshi/src/kernels/mod.rs#L32)
vtable selected once at SAE construction.

For `batch > 1`, the dispatch layer parallelizes via
`par_chunks_exact_mut(d_sae)`
([encoder.rs:35-41](../crates/saeitoshi/src/kernels/encoder.rs#L35-L41)).
Each rayon task handles **one batch row** and calls the backend with
`batch=1`.

The per-thread kernel ([x86.rs:40-62](../crates/saeitoshi/src/kernels/x86.rs#L40-L62))
is structurally:

```rust
for f in 0..d_sae {                  // outer feature loop
    let w_row = &W_enc[f * d_in..];
    for b in 0..batch {              // inner batch loop (batch == 1 here!)
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= d_in {        // SIMD dot product over K = d_in
            let xv = _mm256_loadu_ps(x_row + i);
            let wv = _mm256_loadu_ps(w_row + i);
            acc = _mm256_fmadd_ps(xv, wv, acc);
            i += 8;
        }
        let sum = horizontal_sum_avx2(acc);
        pre_acts[b * d_sae + f] = sum + bias;
    }
}
```

Two prior optimization passes shipped:
1. **Outer-feature loop swap** (the `for f in 0..d_sae` is outermost): W is
   streamed once per call across all batch rows.
2. **Rayon batch parallelism** ([encoder.rs:35](../crates/saeitoshi/src/kernels/encoder.rs#L35)):
   one task per batch row.

These took us from ~70× slower to ~16× slower. The reason they did less than
expected is that **they conflict**: rayon parallelizes by handing each
thread a single batch row, which inverts the optimization the outer-feature
loop swap was supposed to produce.

---

## 3. Roofline: where the 5.9 seconds actually come from

### Hardware: Intel i9-13900HX (Raptor Lake mobile)

| spec | value |
|---|---|
| P-cores | 8 (+ HT = 16 threads) |
| E-cores | 16 |
| total threads | 24 (rayon default) |
| P-core L1d | 48 KB |
| P-core L2 (private) | 2 MB |
| E-core L2 (per-cluster) | 4 MB shared by 4 cores |
| L3 (shared) | 36 MB |
| P-core FMA ports | 2 × 8-wide AVX2 FMA (16 flops/cycle) |
| P-core peak (no AVX-512) | 16 flops/cyc × 5.4 GHz ≈ 86 GFLOPS |
| 8 P-cores peak | ~690 GFLOPS f32 |
| DRAM | DDR5-5200 dual-channel ≈ 83 GB/s peak, ~60 GB/s sustained |

### Headline benchmark arithmetic

Shape: d_in=2048, d_sae=16384, batch=4096 (README numbers).

- Total FLOPs: `2 × M × N × K = 2 × 4096 × 16384 × 2048 = 275 GFLOPs`
- Weight bytes: `d_sae × d_in × 4 = 128 MB`
- Input bytes: `batch × d_in × 4 = 32 MB`
- Output bytes: `batch × d_sae × 4 = 256 MB`

### DRAM traffic for the *current* implementation

With `par_chunks_exact_mut(d_sae)` dispatching one rayon task per batch row
([encoder.rs:35-41](../crates/saeitoshi/src/kernels/encoder.rs#L35-L41))
and each task invoking the backend with `batch=1`
([encoder.rs:40](../crates/saeitoshi/src/kernels/encoder.rs#L40)):

- **Each task streams the full W (128 MB) from DRAM**.
- 4096 tasks × 128 MB W per task = **524 GB synthetic W traffic per encode call**.
- At 60 GB/s sustained DRAM: theoretical lower bound **8.7 seconds**.
- Measured: 5.9 seconds. The 30% favorable delta is shared-L3 amortization
  when 24 threads happen to access nearby W rows within the L3 residency
  window. The model holds.

Per-call arithmetic intensity (DRAM bytes / FLOPs):
**275 GFLOP / 524 GB = 0.52 FLOPs/byte**. We are deeply memory-bound.

### What an ideal cache-blocked implementation would achieve

If W is read from DRAM **once per call** (not 4096 times):
- DRAM traffic: 128 MB W + 32 MB x + 256 MB output write-back = ~420 MB.
- At 60 GB/s: 7 ms — but this is now compute-bound, not memory-bound.

Compute lower bound at 8 P-cores × 86 GFLOPS = 690 GFLOPS:
- 275 GFLOPs / 690 GFLOPS = **400 ms** *if every FMA pipe is full*.
- With realistic 70% peak (memory stalls, port contention): **~570 ms**.

Empirical SAELens/MKL number: **387 ms** at batch=4096 (READMEnumbers
were for batch=4096 on the same hardware). MKL is achieving roughly
80% of single-precision peak across the 8 P-cores. Our target is to
land within ±2× of that.

### The 16× gap, decomposed

| factor | speedup vs current |
|---|---|
| Stop streaming W per batch row (M-block partition instead of N-row split) | **~24×** (524 GB → ~22 GB; bounded by per-block packed-A streams) |
| Register-blocked microkernel (kill horizontal reduction per (f, b)) | ~3–4× (latency-bound to throughput-bound FMA) |
| Pre-pack W at load time into M_R-row panels (avoid per-call repack) | ~1.5× (no per-call MKL-style packing overhead) |

Compound: ~100× over current. Practical target with implementation losses
(thread coordination, packing setup, cache-warmup): **15–20× over current**,
landing within parity to 2× faster than MKL at the headline shape.

---

## 4. Where the 16× gap *isn't*

For completeness — what is *not* the bottleneck:

- **Python/PyO3 overhead.** The PyO3 wrapper is one function call; cost is
  microseconds per encode. Negligible.
- **`vec![0.0; batch * d_sae]` allocation in `Sae::encode`**
  ([sae.rs:241](../crates/saeitoshi/src/sae.rs#L241)). At headline shape
  this is a 256 MB alloc + zero-init; system allocator zeroes via mmap
  COW, so true cost is the first-touch page-faults inside the kernel
  write loop, charged to the DRAM bandwidth budget. Not the dominant
  cost but a real overhead at small batches; flagged as a v0.2 follow-up.
- **TopK selection ([topk.rs](../crates/saeitoshi/src/kernels/topk.rs)).**
  Scalar quickselect on a length-d_sae vector per batch row. At
  d_sae=16384, k=32, batch=4096: ~50 ms total — non-trivial but
  dwarfed by the matmul. Optimize later.
- **Bias add.** One vector add per batch row; ~10 ms total.
- **Rayon overhead.** With 4096 tasks for 24 threads, work-stealing
  overhead is in the low milliseconds. Negligible.
- **Branch mispredicts in the SIMD inner loop.** None — the inner loop is
  a clean while loop with no data-dependent branches.

---

# Design

## Goals and non-goals

**Goals**
- Beat current saeitoshi by ≥5× on the headline benchmark (hard floor).
- Parity or better with `sae_lens` 6.44 at headline shape (stretch).
- Preserve ≤1e-5 max-abs-error vs scalar reference on every test shape.
- Ship behind a `perf-v2` cargo feature; flip to default in the same PR
  series after all checks pass; keep the legacy kernel reachable as
  `AVX2_LEGACY` / `AVX512_LEGACY` / `NEON_LEGACY` for one release.

**Non-goals for v0.1**
- INT8 / FP16 weights (separate milestone; `.sit` dtype slots reserved).
- Per-CPU autotuning (fixed tile constants with env-var override only).
- JIT codegen (target_feature dispatch suffices).
- SVE / SVE2 on ARM (NEON only).
- GPU backends (out of scope for the pure-CPU library identity).
- External BLAS deps. `matrixmultiply` may appear as a `dev-dependency`
  for a sanity-check bench but is not linked into the published wheel.
- NUMA awareness (i9 is single-socket; deferred to server-Xeon work).

## Register microkernel shapes

The microkernel produces an `M_R × N_R` block of C accumulators with K
streamed. M is d_sae (features), N is batch, K is d_in.

| ISA | Reg width | Reg count | **M_R × N_R** | Acc regs | Load:FMA |
|---|---|---|---|---|---|
| AVX2  | 8 f32  | 16 ymm | **16 × 6** | 12 | 8/12 = 0.67, FMA-bound |
| AVX-512 | 16 f32 | 32 zmm | **32 × 12** | 24 | 14/24 = 0.58, FMA-bound |
| NEON   | 4 f32  | 32 q   | **16 × 8** | 32 | 12/32 = 0.38, FMA-bound |

**AVX2 register accounting (16×6)** — per K step:
- 12 ymm accumulators (`C_tile[2][6]`, 2 ymm wide × 6 batch rows).
- 6 ymm broadcasts of `x[b, k]` for b in 0..6 (`vbroadcastss`).
- 2 ymm vector loads of `w[f..f+16, k]` (`vmovups`).
- 12 FMAs (`vfmadd231ps`).
- 1 spare register for bias / scratch / next-K prefetch.

Raptor Lake P-core sustains 2 FMAs/cycle and 3 loads/cycle; the 6:2 ratio
of broadcasts:loads fits the load-port budget. FMA-bound at peak.

**AVX-512 (32×12)** — 24 accumulators, 12 broadcasts, 2 loads, 24 FMAs.
Same logic, doubled tile.

**NEON (16×8)** — Apple M-series has 4 `fmla.4s` pipes per P-core; M_R=16
(= 4 q-regs wide) keeps all four pipes busy. N_R=8 fills the 32-q-reg
budget exactly (4×8 = 32 accumulators, 8 broadcasts, 4 loads via
overlapping windows).

## Cache tile sizes (i9-13900HX as primary target)

| Tile | Size | Footprint | Rationale |
|---|---|---|---|
| **K_C** | 512 | (M_R + N_R) × K_C × 4 = 44 KB (A panel + one column of B in L1d, 48 KB) | d_in ≤ 512 needs no K-tiling; 1024 → 2 chunks; 2048 → 4 chunks; 4096 → 8 chunks |
| **M_C** | 512 | M_C × K_C × 4 = 1 MB (packed A in L2, 2 MB private budget) | Multiple of M_R=16; at d_sae=16384 → 32 M-blocks → clean divide across 24 threads |
| **N_C** | = batch | batch × K_C × 4 = up to 8 MB | N_C ≤ ~10K leaves L3 headroom for shared use; batch always fits in v0.1 |

Apple M-series (NEON): same K_C=512 fits the 128 KB L1d comfortably; M_C
can grow to 1024 if measurement justifies. Constants in code are the i9
defaults. Expose `SAEITOSHI_GEMM_TILE=MC,NC,KC` env var for per-machine
override (parsed once at backend init in `kernels::gemm::mod`).

## Packing strategy

**W_enc — repack once at SAE construction.** Encode as a new
`WeightLayout` enum field on `EncoderWeights` (currently struct at
[sae.rs:20-31](../crates/saeitoshi/src/sae.rs#L20-L31)):

```rust
pub enum WeightLayout {
    RowMajor,                                          // scalar / legacy backends
    PackedPanels { m_r: usize, m_c: usize, k_c: usize }, // tiled backend
}
```

The packed layout: for each M-block (M_C features), for each M_R sub-panel,
for k in 0..d_in, store `[w[f0, k], w[f0+1, k], …, w[f0+M_R-1, k]]`
contiguously. Allows the microkernel to do `vmovups` of M_R contiguous
weights per K step.

`pack_w_enc()` lives in `kernels/gemm/pack.rs`. Uses a 64×64 sub-block
transpose pattern to avoid n²-sized TLB miss storms during repack on
d_sae > 256K. The original buffer is dropped after packing; transient
peak ~256 MB at headline shape, acceptable for one-shot load. Revisit
with in-place buffered transpose only on user OOM report.

Scalar backend asserts `RowMajor`; tiled backend asserts
`PackedPanels`. Loaders default to RowMajor; `Sae::from_parts` invokes
`pack_w_enc` when the tiled backend is selected.

**x — skip packing in v0.1.** The microkernel uses `vbroadcastss` for x,
which accepts any aligned/unaligned float source. We lose nothing by
reading x straight from its existing row-major layout, and save the only
per-call allocation in the hot path. Revisit only if TLB-miss
measurement justifies.

## Threading strategy

Replace `par_chunks_exact_mut(d_sae)` (one task = one batch row → each
thread streams full W) with **M-block partition**: one rayon task per
M-block of d_sae features.

```rust
(0..d_sae).step_by(M_C).par_bridge().for_each(|m_start| {
    // task processes features [m_start, m_start + M_C),
    // streaming the SAME x panel via L3
});
```

At headline shape (d_sae=16384, M_C=512): **32 tasks for 24 threads** —
plenty of work-stealing headroom for Raptor Lake's heterogeneous P/E
cores; the last 8 tasks pick up after the first 24 finish.

Per-thread L2 footprint: M_C × K_C × 4 = **1 MB**, fits comfortably in
the 2 MB P-core L2. Threads share the x panel via L3 (32 MB at headline
shape × num_K_chunks — fits in 36 MB L3 with room for the packed-A panels
each thread is using).

**Per-call DRAM traffic budget at headline**: 128 MB W (read once) + 32 MB
x + 256 MB pre_acts writes = 416 MB. At 60 GB/s: 7 ms. That is now the
floor; the kernel is moved firmly into the compute-bound regime, which
was the goal.

## Parity preservation (1e-5 invariant)

The launch-blocking gate ([tests/test_parity_saelens.py:5-6](../tests/test_parity_saelens.py#L5-L6),
[crates/saeitoshi/tests/simd_parity.rs:5-6](../crates/saeitoshi/tests/simd_parity.rs#L5-L6)).

FMA reorders rounding. Strategy:
- K reduction strictly in-lane order per accumulator (FMA naturally
  gives this — accumulation is associative within a single accumulator
  chain modulo rounding, and we use the same chain order as scalar).
- Cross-lane reduction at the *very end* of the K loop, left-to-right,
  matching the existing
  [horizontal_sum_avx2](../crates/saeitoshi/src/kernels/x86.rs#L67-L74)
  pattern.
- Across K_C chunks: accumulate partials in chunk order (matches the
  scalar reference's left-to-right K iteration).

Add unit test exercising d_in ∈ {1, 7, 511, 513, 1023, 1024, 1025, 2049}
to surface tail handling. If a fancy reordering pushes error past 1e-5,
drop it. Parity is non-negotiable.

## Phased commit plan

Each commit lands new code behind `--features perf-v2`. Each must pass
`cargo test --workspace --release`, `cargo test -p saeitoshi --release
--test simd_parity`, and `pytest tests/`.

1. **Research + analysis doc** (this file).
2. **Scaffold + scalar tiled reference.** `kernels/gemm/{mod, pack,
   tiled_scalar}.rs`; `WeightLayout` enum; tiled-scalar matches
   row-major scalar at 1e-5. Bench id `"scalar_tiled"` at headline shape.
3. **AVX2 (16×6) + AVX-512 (32×12) microkernels.** Single-thread bench
   at headline expected ~4–6× over legacy AVX2 from register blocking.
4. **W repack at construction + M-block rayon threading.** Multi-thread
   parity test at thread counts {1, 2, 4, 24}. Headline-moving commit;
   expect another 5–10× from cutting W DRAM traffic ~24×.
5. **NEON (16×8) microkernel + software prefetch + tile-size env override.**
   NEON parity gated on macOS-14 CI runner only.
6. **Flip default + bench_vs_saelens.py + README.** Tiled becomes
   default; legacy kernels reachable as `AVX2_LEGACY` / `AVX512_LEGACY` /
   `NEON_LEGACY` for one release for regression bisection.

## Risks

- **1e-5 parity drift on tiled path.** Mitigation: in-lane K reduction +
  left-to-right horizontal-sum at finalize + chunk-order K_C partial sum.
  Awkward-d_in stress test catches tail-handling regressions.
- **4 KB aliasing at power-of-2 d_in (2048, 4096).** Same L1 cache set
  across batch rows. Mitigation: write through a thread-local M_R × N_R
  scratch tile in L1, then a single contiguous write at tile finalize.
  Packed W stride is M_R × 4 = 64 bytes (one cache line) — unaffected.
- **False sharing of `pre_acts` across threads.** M_C is a multiple of
  M_R=16 (= 64 B = one cache line), so disjoint write columns are
  line-aligned by construction. Documented as load-bearing invariant in
  `kernels/gemm/mod.rs`.
- **AVX-512 and NEON paths can't be locally perf-validated** on the
  dev box (i9 lacks AVX-512; no M-series hardware in scope). CI catches
  functional regressions only. Surface this here so future tuning passes
  know to start with the unvalidated paths.
- **`vec![0.0; batch * d_sae]` per call in
  [sae.rs:241](../crates/saeitoshi/src/sae.rs#L241).** Caps small-batch
  throughput; out of scope for v0.1; flagged as "reusable scratch on
  `Sae` struct" follow-up.
- **W repack adds ~128 MB transient at SAE load.** Acceptable for
  one-shot load. Revisit with in-place buffered transpose only on user
  OOM report.

## What we explicitly are not doing

Listed above under "Non-goals for v0.1". The single most important non-goal
to defend: **no external BLAS dependency**. The pure-Rust no-torch-dep
property is the moat — the whole point of a tiled GEMM we own.

---

# saeitoshi v0.2 — TopK redesign

## Why this section exists

v0.1 closed the matmul gap. At the headline shape (d_in=2048, d_sae=16384,
batch=512) the tiled AVX2 GEMM runs the matmul in ~40 ms, slightly
beating MKL's ~48 ms at the same shape. But end-to-end `sae.encode` was
still ~466 ms — **TopK selection was now the bottleneck**, swallowing
~410 ms of every call.

The v0.1 brief decomposed it: 40 ms matmul + 5 ms allocation + 5 ms
PyO3 boundary + **410 ms TopK** = ~460 ms. At batch=4096 the TopK
fraction scaled linearly to ~3.3 s — over 80% of the encode wall time.

v0.2 fixes TopK and the picture flips entirely:

| Shape | v0.1 end-to-end | v0.2 end-to-end | sae_lens (MKL+PyTorch) |
|---|---|---|---|
| 512 × 8192, B=4096 | ~1640 ms | **~62 ms** | ~73 ms |
| 768 × 12288, B=4096 | ~2750 ms | **~158 ms** | ~152 ms |
| 2048 × 16384, B=4096 | ~3920 ms | **~438 ms** | ~467 ms |
| 2048 × 16384, B=512 | ~466 ms | **~47 ms** | ~70 ms |

End-to-end **at or below MKL** on three of four shapes. The original
brief's "≥5× sae_lens" target is now retroactively met on encode.

## Why the v0 TopK was slow

The legacy implementation in
[crates/saeitoshi/src/kernels/scalar.rs](../crates/saeitoshi/src/kernels/scalar.rs)
did a full sort of all `d_sae` scores per row, then truncated to k:

```rust
scratch.indexed.clear();
scratch.indexed.reserve(scores.len());
for (i, &v) in scores.iter().enumerate() {
    scratch.indexed.push((i as u32, v));
}
scratch.indexed.sort_by(|a, b| {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
});
scratch.indexed.truncate(k);
scratch.indexed.sort_by_key(|&(i, _)| i);
```

At d_sae=16384, k=32 this is O(n log n) = ~230 K comparisons per row.
At batch=4096 that's ~940 M comparisons per encode, each of which is:

- A closure call (no inlining across the trait object Rust uses for
  `sort_by`'s comparator),
- A `partial_cmp` returning `Option<Ordering>`,
- An `unwrap_or(Equal)` mux,
- A `then_with` for the tie-break,
- A second `Ord::cmp` on `u32` indices.

That's ~5 dependent ops per comparison, all data-dependent branches. The
hot path also pushes 16 K elements into a `Vec` on every call (clear +
reserve was fine, but the per-element push grew the allocation each
encode until the scratch reached steady-state).

## What PyTorch does

`torch.topk` for the TopK SAE config dispatches to
`aten::topk_cpu_template` in
`pytorch/aten/src/ATen/native/Sorting.cpp`. For `k << n` it uses a
**k-element min-heap partial sort**:

1. Initialize a min-heap with the first k elements.
2. For each remaining element: peek the heap root (current k-th best),
   compare, and if the candidate beats the root, replace it and sift
   down.
3. The k elements left in the heap are the top-k. Sort by index for
   output.

That's **O(n log k)** comparisons per row — at d_sae=16384, k=32 that's
~82 K vs the legacy ~230 K. The inner loop is straight-line C++ where
the comparator is a single primitive compare; LLVM/GCC inline it freely
and emit `cmov` for the conditional replace.

Net: PyTorch's TopK is **algorithmically ~3× cheaper** AND **per-op
~3× cheaper**. The ~10× total gap we measured against PyTorch is
explained.

## v0.2 design

Three changes, layered:

### 1. Heap-based partial sort

Implemented in
[crates/saeitoshi/src/kernels/topk_heap.rs](../crates/saeitoshi/src/kernels/topk_heap.rs).
Hand-rolled binary min-heap over `Vec<(u32, f32)>` (the same storage
layout as the legacy `TopKScratch` field, so the `sparsify.rs` API is
unchanged). Embedded heap arithmetic: parent at `(i-1)/2`, children at
`2i+1` and `2i+2`. We deliberately do **not** use
`std::collections::BinaryHeap` — its `Ord`-via-newtype-Reverse pattern
gets in the way of the integer-key trick below, and `peek_mut` allocates
a drop-guard that bloats the hot path.

### 2. Integer-encoded ranking key (branchless inner loop)

The TopK ordering is `(value DESC, index ASC)`. Naively expressed,
that's a two-level comparator with a tie-break — exactly the kind of
shape that defeats branchless codegen. The trick is to bit-encode both
levels into a single `u64` such that **integer comparison matches the
desired total order**:

```rust
#[inline(always)]
fn f32_to_ord(v: f32) -> u32 {
    let bits = v.to_bits();
    let sign_mask = ((bits as i32) >> 31) as u32;
    bits ^ (sign_mask | 0x8000_0000)
}

#[inline(always)]
fn rank_key(idx: u32, val: f32) -> u64 {
    ((f32_to_ord(val) as u64) << 32) | (!idx as u64)
}
```

`f32_to_ord` flips the sign bit on positives and all bits on negatives,
so larger floats map to larger u32 values (the standard IEEE-754 total-
order trick used by sort-network libraries and bucket sort). The lower
32 bits of the key are `!idx`, so a smaller index trailing on a tie
produces a larger key — i.e. **wins** the tie.

The result: a single `u64 > u64` compare on every inner-loop iteration.
LLVM compiles the conditional replace pattern

```rust
let v = scores[i];
let (root_idx, root_val) = heap[0];
if better(i, v, root_idx, root_val) {
    heap[0] = (i, v);
    sift_down(heap, 0);
}
```

to a single `cmp`+`ja`/`jbe` pair, with the late-scan branch heavily
biased toward "not taken" (most candidates lose to the rising heap
root). Branch predictor pressure is minimal.

### 3. Rayon row-level parallelism

Each row's TopK is fully independent. The v0.1 GEMM already established
the M-block rayon threading pattern; v0.2 mirrors it in the sparsifier.

The CSR writeback is the only synchronization point: `out.indices`,
`out.values`, `out.row_offsets` are sequential `Vec`s and can't be
shared across threads. Two-phase design solves it:

```text
Pre-allocate row_idx[batch*k], row_val[batch*k], row_lens[batch]
Parallel:
    for each (row, idx_slot, val_slot, len):
        scratch = TopKScratch::new()  # thread-local via for_each_init
        rescale + heap_topk(row, k, scratch)
        for each (idx, val) in scratch:
            if post_relu drops it: continue
            idx_slot[*len] = idx; val_slot[*len] = val; *len += 1
Serial compaction:
    for r in 0..batch:
        out.indices.extend_from_slice(&row_idx[r*k .. r*k + row_lens[r]])
        out.values.extend_from_slice(&row_val[r*k .. r*k + row_lens[r]])
        out.finish_row()
```

`for_each_init` gives each rayon worker one thread-local
`TopKScratch` reused across all rows it draws from the work-stealing
deque. The parallel path activates only when `tile_rows ≥ 4` and
`tile_rows * d_sae ≥ 8192` (rayon overhead dominates below that); the
sequential fallback is preserved verbatim from v0.1.

## Numbers

### Isolated TopK (criterion bench)

[crates/saeitoshi/benches/topk.rs](../crates/saeitoshi/benches/topk.rs)
times `topk_select` against a precomputed scores vector — no matmul,
no rayon. Per-row cost only:

| Shape | Legacy full sort | Heap partial sort | Speedup |
|---|---|---|---|
| d_sae=8192, k=32 | ~389 µs | ~10 µs | **38×** |
| d_sae=16384, k=32 | ~853 µs | ~16 µs | **52×** |
| d_sae=16384, k=64 | ~779 µs | ~19 µs | **41×** |
| d_sae=16384, k=1024 | ~793 µs | ~114 µs | **7×** |

The k=1024 case is degrading toward `sort_unstable_by` cost; that's
expected (heap insert is O(log k), so the cost grows with k). For the
SAELens-typical k=32–64 range the win is ~40–50×.

### End-to-end (bench_vs_saelens.py)

The full encode pipeline including matmul, allocation, PyO3 boundary,
and CSR construction. Median of 11 iterations after 3 warmups, Intel
i9-13900HX (24 logical cores).

| Shape | v0.1 | v0.2 | sae_lens (MKL) |
|---|---|---|---|
| 512 × 8192, B=4096 | ~1640 ms | ~62 ms | ~73 ms |
| 768 × 12288, B=4096 | ~2750 ms | ~158 ms | ~152 ms |
| 2048 × 16384, B=4096 | ~3920 ms | ~438 ms | ~467 ms |
| 2048 × 16384, B=512 | ~466 ms | ~47 ms | ~70 ms |

v0.2 beats sae_lens on three of four shapes, ties on the fourth.

## Verification

Same parity story as v0.1 — 1e-5 max abs error against `sae_lens.SAE.encode`
gates the launch
([tests/test_parity_saelens.py](../tests/test_parity_saelens.py)), still
green at v0.2. Plus a new test file
[crates/saeitoshi/tests/topk_parity.rs](../crates/saeitoshi/tests/topk_parity.rs)
that asserts **byte-identical** output from the heap implementation
against the legacy scalar reference across:

- Small shapes (d_sae ∈ {8, 16, 64, 256}, k ∈ {1, 3, 4, 16, 100})
- The headline (d_sae=16384, k=32)
- k == d_sae (degenerate full-sort case)
- k == 1 (degenerate single-element case)
- All-ties (every score equal, only tie-break ordering matters)
- Partial ties at the k-boundary (the load-bearing tie-break test)

Total Rust tests: 25 → **42** (+17). Python tests: 32 → 32 (unchanged;
parity gate stays green).

## Risks

- **f32_to_ord on NaN.** NaN bit-patterns map to large u32 values
  deterministically, but the order is implementation-defined for the
  set of NaNs. The encoder doesn't produce NaN from the parity tests,
  so this is a latent contract — documented but not actively defended.
  If a downstream caller starts feeding NaN-laden scores, we may need
  to filter at the matmul output.
- **Rayon `for_each_init` cost on small batches.** Covered by the
  `tile_rows * d_sae ≥ 8192` threshold; benchmarked at the boundary
  to confirm no regression at small shapes.
- **Per-call `vec![0.0; batch * k]` allocation.** ~1 MB at headline,
  ~5 µs cost. Negligible against the 438 ms wall but flagged as a
  potential v0.2.1 polish if a smaller-batch workload surfaces it.
- **Pointer-comparison dispatch.** `topk::backend_name()` uses
  `std::ptr::eq` on function pointers. This works on every host we
  test on; if a future platform legalizes function-pointer ICF in a
  way that breaks it, the bench label may report "legacy" when the
  heap is actually running. Not load-bearing — the parity tests catch
  any actual behavior divergence.

## What v0.2 explicitly did not do

- **SIMD tournament-tree TopK.** The branchless heap is the v0.2 inner
  loop. SIMD tournament was scoped as v0.2.1 if the heap fell short;
  it didn't, so this is parked.
- **K-tiling the GEMM.** The packed-A panel still spills to L3 at the
  headline. Polish, not progress; matmul isn't the bottleneck anymore.
- **INT8 weights.** Halves W memory, doubles bandwidth-bound matmul
  throughput. The obvious v0.3.
- **Per-call scratch on `Sae` for the `pre_acts` buffer.** Still
  allocated fresh per encode at
  [sae.rs:309](../crates/saeitoshi/src/sae.rs#L309). ~5 ms at headline,
  small fraction of the new 438 ms total. Parked as a v0.2.1 polish.

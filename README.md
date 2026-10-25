# mini-inference-engine

A neural-network **inference engine written from scratch in modern C++** — no deep-learning
libraries, no BLAS. It loads an MLP trained in PyTorch, recognizes MNIST handwritten digits,
and reproduces PyTorch's output (same prediction, logits agreeing to several significant
figures). The matrix-multiply (GEMM) core is
then optimized **from ~1.5 to ~300 GFLOPS (~200×)** on an Apple M4 Pro, with every step measured.

> Built to deepen my C++ / systems / performance-engineering fundamentals. The interesting part
> is not that it runs — it's the measured optimization journey and the things the data proved wrong.

---

## Highlights

- **From scratch.** Own `Tensor` (manual heap memory, strides, full Rule of Five), operators
  (GEMM, ReLU, Linear/FC, Softmax), binary weight loader — **zero deep-learning dependencies**.
- **Correct.** Feed a 28×28 MNIST image → the engine predicts the digit, and its logits match
  PyTorch's (not just the argmax — the actual numbers line up, to several significant figures).
- **Fast.** The GEMM core goes from ~1.5 GFLOPS (naive) to ~300 GFLOPS through cache-aware loop
  order, SIMD (ARM NEON), and multithreading — each step benchmarked and explained.

## GEMM optimization (N = 2048, Apple M4 Pro, `-O3`)

| Version | GFLOPS | Speedup | What changed |
|---|---|---|---|
| Naive triple loop (scalar) | ~1.5 | 1× | baseline; strided access, falls off a cache cliff at large N |
| Cache-friendly order + SIMD | ~32 | ~21× | `ikj` loop order → sequential access, then `__restrict__` unlocks compiler NEON auto-vectorization (4 floats/op) |
| Multi-threaded | ~300 | ~200× | split output rows across cores (10 P-cores); also a hand-written thread pool |

*Benchmarks use warm-up + minimum-of-5 runs, verify correctness against the naive result, and
sweep N = 512/1024/2048. See `src/bench.cpp`.*

## How it works

```
28×28 image ──▶ Tensor ──▶ Linear(784→128) ──▶ ReLU ──▶ Linear(128→10) ──▶ Softmax ──▶ argmax ──▶ digit
```

- `Tensor` — N-dimensional array over a raw heap buffer with row-major strides; owns its memory
  by hand (RAII + Rule of Five) to practice and demonstrate C++ resource management.
- Operators live in `ops.*` (ReLU, Linear, Softmax) and `gemm.*` (matmul, with selectable
  Naive / Reorder / Blocked / Threaded / ThreadedPool modes).
- Weights come from `python/export.py`: it trains the MLP in PyTorch and dumps each tensor as a
  raw `float32` binary; C++ reads them back with `std::ifstream` (see `loader.*`). PyTorch's
  `[out, in]` weight layout is transposed on export to match the C++ GEMM.

## Build & run

Requires a C++20 compiler and CMake ≥ 3.20.

```bash
# 1) generate the trained weights + a test image (one-time; needs Python + PyTorch)
pip install torch torchvision numpy
python python/export.py          # writes model/*.bin

# 2) build and run the engine
cmake -B build
cmake --build build
./build/main                     # prints the logits and the predicted digit

# optional: run the GEMM benchmark
./build/bench
```

## The optimization journey (what the data taught me)

- **Profile before optimizing.** The first version used a convenient indexing API that heap-allocated
  a small vector *on every element access* — 512³ matmul took **33 s**. Switching to flat indexing
  dropped it to **0.37 s** (~100×). The bottleneck was an abstraction cost, not the algorithm.
- **Cache behavior depends on size.** `ikj` loop reordering only beat the naive order once the
  working set exceeded the 16 MB L2 cache — I found the crossover by sweeping matrix sizes, not by
  trusting the textbook.
- **SIMD, the easy way.** Rather than hand-writing NEON intrinsics, I used `__restrict__` to prove
  to the compiler that the matrices don't alias, which unlocked its auto-vectorizer (confirmed with
  `-Rpass=loop-vectorize`). One keyword → ~20×.
- **Measure, don't assume.** Cache blocking gave *no* improvement on this hardware, and a thread
  pool was no faster than spawning threads per call for a single large matmul. Both are "better
  engineering" in general, but the data showed they don't help *this* workload — a reminder that
  optimizations are workload- and hardware-specific.

## Limitations & next steps

- CPU only for now; a CUDA GEMM (naive → shared-memory tiled → vs cuBLAS) is the planned next step.
- `float32` only; real engines add more dtypes and quantization (int8).
- Single model (MNIST MLP); no operator fusion, batching, or memory planning that production
  engines (ONNX Runtime, llama.cpp) do.

## Tech

C++20 · CMake · std::thread / mutex / condition_variable / latch · ARM NEON (auto-vectorized) ·
PyTorch (training/export only).

---

*Author: Ziqi Tian*

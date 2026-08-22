# Celeris — Design

## Purpose

Celeris implements core linear algebra operations (vectors and matrices) in Rust, accelerated
using AVX2 SIMD intrinsics via `std::arch`, and benchmarks the result against both a naive
scalar Rust implementation and NumPy — proving not just that SIMD is faster than a naive
approach, but that it's competitive with an established, widely-used numerical library.

This project exists primarily as a hands-on learning vehicle for hardware-level performance
work (SIMD, CPU architecture, benchmarking methodology) grounded in linear algebra fundamentals,
ahead of longer-term goals in scientific computing / HPC.

## Architecture

```
[Naive Rust]         [SIMD Rust (AVX2)]         [NumPy (Python)]
      \                      |                        /
       \                     |                       /
        \                    v                      /
         -----------> [criterion benchmarks] <------
                              |
                              v
                     [results across a sweep
                      of input sizes]
```

Three independent implementations of the same operations are built and benchmarked against
each other:

1. **Naive Rust** — straightforward scalar loops, no optimization. The baseline.
2. **SIMD Rust** — AVX2 intrinsics via `std::arch`, operating on multiple values per instruction.
3. **NumPy** — the external reference point, run and timed in its native Python environment.

## Components

### vector

Core vector operations: addition, scaling, dot product, and other standard vector arithmetic.
Implemented three times per the architecture above (naive, SIMD, and the NumPy reference script).

### matrix

Core matrix operations: multiplication, transpose, and other standard matrix arithmetic.
Same three-way implementation pattern as vector operations.

### simd

Internal wrappers around AVX2 intrinsics (`std::arch::x86_64`), isolating unsafe SIMD code
behind a safe interface used by the vector and matrix modules.

### benches

`criterion`-based benchmarks comparing naive Rust, SIMD Rust, and NumPy across a sweep of
input sizes (not a single fixed size), to observe how performance characteristics change with
scale — this sweep data is also what a future GPU-dispatch feature would rely on to find a
CPU/GPU crossover point.

### scripts

Python/NumPy reference implementations used only for benchmark comparison — not a dependency
of the Rust core itself.

## Design decisions

- **`std::arch` over `std::simd`:** `std::simd` (Rust's portable SIMD API) is nightly-only.
  Targeting stable Rust means using `std::arch` intrinsics directly, which are more verbose
  and platform-specific, but require no special toolchain setup for anyone building the project.
- **AVX2 over SSE/AVX-512:** AVX2 offers a strong balance of being meaningfully modern (256-bit
  registers, real throughput gains) while remaining broadly supported on hardware from roughly
  the last decade — unlike AVX-512, which has inconsistent support across chips.
- **`f64` only for MVP:** `f64` is the more common default in general-purpose/scientific
  numerical computing (matches NumPy's own default dtype), and precision matters more than
  raw throughput for this project's framing. `f32` is deferred to post-MVP.
- **Fixed-size, stack-allocated vectors/matrices (const generics):** Vector and matrix sizes are
  fixed at compile time via Rust's const generics, backed by stack-allocated arrays rather than
  heap-allocated `Vec`. This avoids per-operation heap allocation/deallocation overhead and the
  pointer indirection of heap-backed storage. Trade-off: no support for runtime-determined
  (dynamically-sized) vectors or matrices. Stack space is finite and imposes a practical ceiling
  on supported sizes; the exact safe bound (accounting for multiple live operands per operation,
  not just a single buffer in isolation) is not yet determined.

## Non-goals (for MVP)

- Dynamically-sized (runtime-determined) vectors/matrices — sizes are fixed at compile time via
  const generics
- `f32` support (post-MVP)
- GPU execution or CPU/GPU dispatch (post-MVP)
- Multi-threading beyond SIMD (post-MVP)
- Sparse matrix support
- Higher-level operations (solving linear systems, eigenvalues, etc.)
- Python bindings for the Rust core itself

## Post-MVP roadmap

- **Heuristic/adaptive CPU-GPU dispatch:** benchmark a GPU implementation against the SIMD CPU
  core to empirically find the input-size crossover point where GPU execution wins despite data
  transfer overhead, then route operations to whichever backend is faster based on that data.
- **`f32` support** alongside `f64`, including benchmarking the precision/speed tradeoff between them.
- **Multi-threading (Rayon)** layered on top of SIMD, parallelizing across cores in addition to
  within them.
- **CLI or visualization tool** to chart benchmark sweep data rather than only printing numbers.
- **Sparse matrix support** — a distinct linear algebra topic using different algorithms than
  the dense operations built for MVP.
- **Higher-level LA operations** (solving linear systems, eigenvalues) built on top of the
  vector/matrix primitives.
- **Python bindings via PyO3**, allowing the Rust core to be called from Python — a direct
  parallel to how NumPy itself works under the hood (a compiled, fast core exposed to Python).

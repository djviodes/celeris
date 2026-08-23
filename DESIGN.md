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

## Correctness verification

Beyond the performance benchmarks described above, naive Rust, SIMD Rust, and NumPy results are
checked against each other for correctness, not just speed:

- **Golden-value tests:** a set of hand-calculated example inputs/outputs, checked against all
  three implementations. Catches a bug shared across all three (e.g. a misunderstood operation)
  that comparing implementations against each other alone wouldn't reveal.
- **Oracle/differential testing:** a larger set of randomly generated inputs, checked for
  agreement across naive, SIMD, and NumPy, using `proptest` for input generation and shrinking.
  Catches an implementation that diverges from the other two.
- **Floating-point comparison:** results are compared using combined absolute + relative
  tolerance rather than exact equality. SIMD execution (fused multiply-add, parallel-lane
  accumulation order) can produce results that differ from naive scalar execution in the last few
  bits even when both are correct, since floating-point arithmetic isn't strictly associative.
  Pure relative tolerance (and ULP-based comparison) degrade near zero; the combined approach
  avoids that failure mode.

NumPy is invoked from the Rust test suite as a subprocess (`std::process::Command`), passing
inputs as arguments and reading the result back from the script's output, rather than embedding
Python via PyO3. This keeps Python entirely out of the Rust core's build (consistent with Python/
NumPy being a benchmark-comparison-only dependency, not a runtime dependency of the core, per
README) — subprocess-spawn cost is irrelevant here since this only runs during `cargo test`, not
inside anything benchmarked.

## SIMD remainder handling and memory alignment

AVX2 processes 4 `f64` per register, and vector/matrix sizes (fixed via const generics) won't
always be a multiple of 4. Storage is padded up to the next multiple of 4 by default, so
operations always work on full registers with no separate scalar tail loop. The padding value at
rest is chosen to be a safe identity element for addition/multiplication-based operations (e.g.
`0`, which doesn't affect a sum, dot product, or scale).

Not every operation is safe with that resting padding value — the test is whether the padding
value matches the operation's own mathematical identity element. Where it doesn't (e.g. `min`,
whose identity element is `+infinity`, not `0`), masked SIMD operations are used instead of
relying on the default padding. This is decided per-operation, not as a single universal rule.

Storage uses `#[repr(align(32))]` to guarantee 32-byte alignment, matching AVX2's 256-bit
register width, so aligned load/store instructions (`_mm256_load_pd`/`_mm256_store_pd`) are used
rather than their unaligned counterparts. Combined with the padding above (storage always sized
to a multiple of 32 bytes), every SIMD register-width chunk in the array stays 32-byte aligned,
not just the first one.

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

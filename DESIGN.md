# Celeris — Design

## Purpose

Celeris implements core linear algebra operations (vectors and matrices) in Rust, accelerated
using AVX2 SIMD intrinsics via `std::arch`, and benchmarks the result against both a naive
scalar Rust implementation and NumPy — proving not just that SIMD is faster than a naive
approach, but that it's competitive with an established, widely-used numerical library.

This project exists primarily as a hands-on learning vehicle for hardware-level performance
work (SIMD, CPU architecture, benchmarking methodology) grounded in linear algebra fundamentals,
ahead of longer-term goals in scientific computing / HPC — including eventual use as the
numerical core for a separate physics simulation platform, which shapes some operation-scope
decisions below (e.g. cross product).

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

Core vector operations: addition, subtraction, scaling, dot product, outer product (produces a
matrix), Euclidean (L2) and Manhattan (L1) norms (defined for any size `N`), and other standard
vector arithmetic — exact remaining scope still being finalized (see Design decisions). Cross
product is a planned early post-MVP addition, not MVP (see Design decisions and roadmap).
Implemented three times per the architecture above (naive, SIMD, and the NumPy reference script).

### matrix

Core matrix operations: addition, subtraction, scaling, matrix-matrix multiplication,
matrix-vector multiplication, transpose, determinant (1×1, 2×2, and 3×3 only for MVP — see
Design decisions), Frobenius norm, matrix 1-norm (max absolute column sum), and matrix ∞-norm
(max absolute row sum) — all defined for any size — and other standard matrix arithmetic — exact
remaining scope still being finalized. Same three-way implementation pattern as vector operations.

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

## Error handling

Because vector/matrix sizes are fixed at compile time via const generics, most "mismatched
dimensions" scenarios (e.g. adding a `Vector<3>` to a `Vector<4>`, or multiplying incompatible
matrix shapes) are type errors caught by the compiler — there's no runtime handling needed for
them at all.

Two genuinely runtime-fallible cases remain, handled differently depending on whether the
failure is a programmer logic error or genuinely external/untrusted data:

- **Indexing with a runtime-computed index:** returns `Option<&T>` (matching `slice::get`'s
  convention in Rust's standard library). Since a `Vector<N>`'s length is a compile-time
  constant, an out-of-range index almost always reflects a bug in the calling code, not
  untrusted external input — there's only one possible failure reason, so no additional error
  context is needed.
- **Constructing a fixed-size vector/matrix from runtime-length external data** (e.g. NumPy
  subprocess output, `proptest`-generated inputs, eventually real user input): returns `Result`,
  via a `TryFrom` implementation, carrying a descriptive error (e.g. expected vs. actual length).
  This data genuinely originates outside the program's control, so the caller needs context to
  understand and handle the failure, per Rust API Guidelines: don't panic on invalid external
  input, return `Result` instead.

## Design decisions

- **Column-major matrix storage:** Neither row-major nor column-major is inherently faster —
  cache-friendliness depends on whether traversal order matches the storage layout, and this cuts
  both ways symmetrically (row-major favors row-wise access, column-major favors column-wise
  access). Matrix multiplication inherently requires walking one operand row-wise and the other
  column-wise regardless of which single layout is chosen for both, so this decision doesn't
  resolve that on its own — it requires separate techniques (transposing an operand, cache
  blocking) applied independently of the storage layout. Column-major matches how the author
  naturally reasons about linear algebra (columns as the primary unit — e.g. matrix-vector
  multiplication as a linear combination of columns) and matches the historical convention used
  by Fortran/BLAS/LAPACK, the foundation underneath much of scientific computing software. Note:
  NumPy's own default array layout is row-major ("C order"), even though it delegates much of its
  heavier linear algebra internally to column-major BLAS/LAPACK — this is a storage-convention
  difference from the benchmark comparison target only, not a correctness concern.

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
- **AVX2 hardcoded at compile time for MVP, no runtime detection:** MVP targets known hardware
  (the author's own machine, confirmed AVX2-capable), so CPU feature detection and multi-
  instruction-set dispatch would add complexity with no benefit yet. AVX2 is isolated behind the
  `simd` module's safe interface (see Components), so runtime detection and dispatch across
  multiple instruction sets can be added later without restructuring the vector/matrix modules
  that call into it. Deferred to post-MVP — see roadmap.

- **Determinant limited to 1×1, 2×2, and 3×3 for MVP:** direct, decomposition-free closed-form
  formulas exist only at these sizes (2×2: `ad - bc`; 3×3: cofactor expansion/rule of Sarrus).
  Beyond 3×3, cofactor expansion scales factorially (~N! terms) and becomes computationally
  intractable — general N×N determinant requires a decomposition-based algorithm (e.g. LU),
  which is already out of MVP scope (see Non-goals). Because the formula itself differs per exact
  size, this is implemented per-specific-N rather than as one generic function over
  `Matrix<N, N>`, the same shape of constraint as cross product being 3D-only. General N×N
  determinant is deferred to post-MVP (see roadmap).

- **Min column/row sum excluded from norms:** NumPy exposes these via `linalg.norm`'s `ord=-1`
  (min column sum) and `ord=-inf` (min row sum), but they don't satisfy the triangle inequality
  (counterexample: two matrices each with min column sum 2 can sum to a matrix with min column
  sum 22), so they aren't mathematically norms despite NumPy's naming convenience. Excluded from
  Celeris's norm operations for MVP; deferred to post-MVP as a separately-named utility, not
  documented as a norm (see roadmap).

- **Cross product deferred to (early) post-MVP:** mathematically relevant to Celeris's longer-term
  goal of powering a physics simulation platform, so it will be added — but it structurally
  doesn't fit MVP's barebones scope the way the rest of the operation list does: it only exists
  for 3D vectors (breaking the generic-`N` pattern every other MVP operation follows), can't
  participate in the size-sweep benchmarking MVP is built around, and is too small an operation
  (3 elements) for SIMD to show a meaningful speedup on. Prioritized as an early post-MVP
  addition given its importance to that future consumer, rather than lumped in with the rest of
  the post-MVP backlog.

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

- **Cross product (3D vectors only)** — early-priority post-MVP addition, ahead of the rest of
  this list, needed for the planned physics simulation platform; deferred from MVP due to its
  fixed-size, non-generic nature (see Design decisions).
- **Dual-layout matrix storage (row-major + column-major) with cached conversions:** maintain
  both layouts for matrices where different operations benefit from different access patterns,
  using Mnemosyne (a separate caching library project, Rust rewrite also post-MVP) to cache
  converted copies rather than re-converting repeatedly. Depends
  on: the matrix mutability model (immutable values sidestep cache invalidation entirely; in-place
  mutation would need an invalidation mechanism), the memory cost of maintaining two
  representations (relevant given the still-unresolved stack-size ceiling from the sizing
  decision), and empirical per-operation benchmark data (see Benchmark methodology, not yet
  decided) to know which operations actually benefit from which layout.
- **Runtime CPU feature detection with per-instruction-set dispatch** (AVX2, AVX-512, etc.),
  replacing MVP's hardcoded AVX2 assumption, so the crate can run correctly (with graceful
  degradation or best-available instruction set) on hardware other than the author's own —
  relevant if this ever moves beyond a personal learning project toward wider distribution.
  The MVP decision to hardcode AVX2 was made on intuition, not research — research still needed
  before implementing this item:
  - `is_x86_feature_detected!` — how it works, when the check happens, and its cost.
  - `#[cfg(target_feature = "avx2")]` and the `-C target-feature`/`-C target-cpu` rustc flags.
  - What happens if a compile-time-assumed feature turns out to be wrong at runtime.
  - `#[target_feature(enable = "avx2")]` — what it does, and why functions calling AVX2
    intrinsics need to be marked with it.
  - How `#[target_feature]` relates to `unsafe`.
  - How existing Rust SIMD-oriented crates handle runtime dispatch between multiple
    implementations.
- **Heuristic/adaptive CPU-GPU dispatch:** benchmark a GPU implementation against the SIMD CPU
  core to empirically find the input-size crossover point where GPU execution wins despite data
  transfer overhead, then route operations to whichever backend is faster based on that data.
- **`f32` support** alongside `f64`, including benchmarking the precision/speed tradeoff between them.
- **Multi-threading (Rayon)** layered on top of SIMD, parallelizing across cores in addition to
  within them.
- **CLI or visualization tool** to chart benchmark sweep data rather than only printing numbers.
- **Sparse matrix support** — a distinct linear algebra topic using different algorithms than
  the dense operations built for MVP.
- **Higher-level LA operations** (solving linear systems, eigenvalues, general N×N determinant
  beyond MVP's 3×3 cap) built on top of the vector/matrix primitives.
- **Min column/row sum utility functions** (matching NumPy's `linalg.norm(ord=-1)`/`ord=-inf`) —
  not true norms (fail the triangle inequality), so named and placed separately from Celeris's
  actual norm operations rather than alongside them; exact API placement not yet decided.
- **Python bindings via PyO3**, allowing the Rust core to be called from Python — a direct
  parallel to how NumPy itself works under the hood (a compiled, fast core exposed to Python).

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

Celeris is organized as a Cargo workspace with (at least) two crates: the numerical core library
(vector/matrix/simd, below) and a separate analysis/benchmarking tool crate — see Design
decisions for why this is split rather than one crate. Exact crate names not yet decided.

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
of the Rust core itself. Also home to the `pyperf`-based NumPy benchmarking script (see
Benchmark methodology), run as its own standalone Python process.

### analysis (separate crate, in Rust)

Combines `criterion`'s Rust-side benchmark output with the Python-side `pyperf` output into one
comparison report. See Benchmark methodology for the MVP vs. post-MVP shape of this tool, and
Design decisions for why it's a separate crate rather than part of the core library.

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

## Benchmark methodology

NumPy is benchmarked as a standalone Python process using Python's own timing tools, not spawned
per-timed-call from Rust (unlike the subprocess mechanism used for correctness checks — see
Correctness verification). Spawning a fresh process per timed call would bake process-spawn
overhead into NumPy's measured performance, invalidating the comparison. Results are written out
as JSON and combined with Rust's `criterion` results into one unified sweep report.

Layout is treated as an accepted, uncontrolled difference between Celeris (column-major) and
NumPy (row-major) rather than something to normalize away — no layout conversion happens for the
benchmark comparison, and no row-major variant of Celeris is built solely to isolate the
variable. This matches the project's actual goal (proving SIMD is competitive with an
established library end-to-end, per Purpose), not an isolated experiment on memory layout itself.

Identical input data across naive Rust, SIMD Rust, and NumPy at each sweep size: `criterion`'s
`bench_with_input` generates each size's input once, outside the timed closure, so naive and
SIMD share identical data within the Rust benchmark. That same generated data is then written
out via the `npyz` crate in NumPy's native `.npy` binary format, which the standalone Python
benchmark script reads directly via `numpy.load()` — avoiding a custom parsing step on the
Python side. `npyz` was chosen over `ndarray-npy` specifically because Celeris doesn't use the
`ndarray` crate anywhere (see Design decisions: fixed-size, stack-allocated vectors/matrices) —
`ndarray-npy` would require adding `ndarray` as a new dependency solely for this export step,
while `npyz` works directly from a raw `Vec<f64>`/slice, which is what Celeris's own storage
would need to be converted to for export regardless of which library was used.

Benchmark hardware/environment recording: on the Python side, `pyperf` (chosen over
`pytest-benchmark`, which is built as a pytest plugin rather than a standalone tool) captures
this by default — CPU model, core count, frequency, OS/platform, memory, load average, and more.
On the Rust side, `criterion` doesn't capture this itself, and research confirmed no existing
Rust crate bundles both a benchmarking engine and system metadata capture the way `pyperf` does
— the established pattern is pairing a benchmarking crate with `sysinfo` for this. So a small
separate module handles it: `sysinfo` for general CPU/system info, and `raw-cpuid` for precise
CPU feature-flag confirmation (directly answering whether AVX2/FMA are actually available on the
benchmarking machine — more precise than anything `pyperf` itself offers, since it queries the
CPU's feature bits directly). This module runs alongside `criterion`'s benchmarks rather than
replacing them — `criterion` already handles the actual measurement engine (calibration,
warm-up, statistics) well; the gap was narrowly about metadata, not the measurement core, so no
reason to duplicate that engine. (Two adjacent tools surfaced during research but don't apply:
`divan`, a newer `criterion` alternative with a simpler API, doesn't capture system metadata
either; `iai`/`iai-callgrind` measures CPU instructions via Cachegrind rather than wall-clock
time — a different, complementary measurement philosophy, not a replacement for this comparison.)

**Combining `criterion` and `pyperf` output — MVP vs. post-MVP shape.** `criterion`'s only
version-stable output is `raw.csv` (its JSON files — `estimates.json`, `sample.json`,
`tukey.json` — are explicitly documented as private and can change without warning); `pyperf`
writes JSON with raw per-sample `values`. Rather than forcing either tool to conform to the
other's format upstream, both benchmarking tools record natively, and a separate Rust crate (see
Components: analysis) does whatever conversion is needed at comparison time:

- **MVP (minimal, narrow, manual):** a small script reading the specific known CSV columns and
  JSON fields this project's specific benchmarks produce (no general-purpose/robust parsing of
  arbitrary future formats). Presents each tool's own already-computed summary statistic (e.g.
  each one's reported mean) side by side, accepting that they're computed via slightly different
  methodologies as one more uncontrolled difference (the same move already made for row-major vs.
  column-major layout) rather than recomputing matched statistics from raw samples. Prints both
  tools' metadata blocks one after another rather than unifying them into one schema.
- **Post-MVP (general, automated, rigorous):** robust parsing that isn't tied to today's specific
  benchmark set, statistics recomputed identically from both tools' raw per-sample data
  (`raw.csv`'s `sample_measured_value`, `pyperf`'s `values`) for a true 1-to-1 methodological fit
  rather than trusting each tool's own summary computation, and a unified metadata schema. This
  extends/replaces the existing "CLI or visualization tool" roadmap item (see roadmap).

**Sweep size ceiling.** Confirmed `criterion` benchmarks run on the main thread (not a spawned
thread), so the applicable stack budget is the OS-level main-thread limit — confirmed via
`ulimit -s` at 8192 KiB (8 MiB) on the author's machine — rather than Rust's smaller 2 MiB
spawned-thread default. Accounting for 3 simultaneous live buffers per operation (e.g. matrix
multiply's two inputs plus one output — the same conservative count applied to both vectors and
matrices below):
- **Matrix sweep ceiling: N=500** (500×500). `500² × 8 bytes × 3 buffers ≈ 5.72 MiB`, leaving
  ~2.28 MiB headroom out of 8 MiB for call-frame overhead, `criterion`'s own internal state, etc.
- **Vector sweep ceiling: N=250,000.** Chosen to exactly match the matrix ceiling's total element
  count (500² = 250,000), since a matrix's elements scale as N² while a vector's scale as N —
  this gives vectors the identical byte usage and headroom as the matrix case, rather than
  needlessly capping vectors at 500 when the same budget supports far more.

This was a provisional, calculation-based ceiling, not empirically verified — deliberately
reducing the stack limit and testing for an actual overflow (e.g. via `ulimit -s` or
`std::thread::Builder::stack_size`) would give a measured rather than calculated number; worth
revisiting once real operation code exists.

## Design decisions

- **Cargo workspace with separate crates for the numerical core and the analysis/benchmarking
  tool, decided for MVP (not deferred):** this is the standard, idiomatic pattern in Rust for a
  core library plus supporting tooling that isn't part of its public API (sometimes called the
  "xtask" pattern) — not over-engineering for this shape of problem. Concretely, it keeps the
  core library's dependency footprint lean: the analysis tool needs CSV/JSON-parsing
  dependencies the core library has no reason to carry, which matters because the core library is
  meant to be consumed by other projects (see Purpose: the planned physics simulation platform)
  that would otherwise transitively inherit those dependencies for no reason. Decided for MVP
  rather than deferred specifically because — unlike other MVP/post-MVP splits in this document —
  this is an organizational choice, not new capability: since both crates are being written for
  the first time regardless, setting up the workspace now costs little, while retrofitting it
  after code already exists tangled together in one crate would mean real refactoring later.

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
- **General, automated, statistically-matched analysis tool** (see Benchmark methodology) —
  robust parsing not tied to today's specific benchmark set, statistics recomputed identically
  from both `criterion`'s and `pyperf`'s raw per-sample data, unified metadata schema, and a CLI
  or visualization layer to chart sweep data rather than only printing numbers. Replaces MVP's
  minimal side-by-side version.
- **Continuous benchmarking / CI regression tracking via Bencher**, layered on top of `criterion`
  (which remains the actual measurement engine) to track performance across commits/PRs over
  time and catch regressions in CI — distinct from the single-run visualization tool above.
  Bencher has both self-hosted and hosted-SaaS options; which to use isn't decided.
- **Sparse matrix support** — a distinct linear algebra topic using different algorithms than
  the dense operations built for MVP.
- **Higher-level LA operations** (solving linear systems, eigenvalues, general N×N determinant
  beyond MVP's 3×3 cap) built on top of the vector/matrix primitives.
- **Min column/row sum utility functions** (matching NumPy's `linalg.norm(ord=-1)`/`ord=-inf`) —
  not true norms (fail the triangle inequality), so named and placed separately from Celeris's
  actual norm operations rather than alongside them; exact API placement not yet decided.
- **Python bindings via PyO3**, allowing the Rust core to be called from Python — a direct
  parallel to how NumPy itself works under the hood (a compiled, fast core exposed to Python).

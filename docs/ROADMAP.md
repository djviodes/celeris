# Celeris — Roadmap

Celeris began as a small SIMD-accelerated linear algebra library. Its scope has grown into a
layered numerical computing core: dense and sparse linear algebra, CPU and GPU backends, a
cross-backend validation harness, language interop, and simulation workloads built on top. This
document is the authoritative description of those layers. [DESIGN.md](../DESIGN.md) covers
architecture and design decisions; [VALIDATION.md](VALIDATION.md) covers how correctness is
established.

No dates or timeline are committed here. Layers ship independently and in the order described
under [Ordering](#ordering-and-dependencies); the order may change as work reveals new
information.

## Status tags

Every item in this repository's roadmap carries exactly one of these tags, and a tag only
describes what exists in the repository today:

- **DONE** — implemented, tested, and merged.
- **IN PROGRESS** — work has started; some of it exists, the rest does not.
- **PLANNED** — not started. Scope and design may still change.

No benchmark results, speedups, or test-coverage figures are claimed anywhere in these docs
unless the corresponding benchmark or test exists in the repository.

## Snapshot

Taken against commit `28e919b`. At that commit the library does not compile, because the
work-in-progress `simd::vector::add` is unfinished; the last commit at which the whole workspace
built and passed its tests is `efaedcc` (59 passing tests).

| # | Layer | Status | Depends on |
|---|---|---|---|
| 1 | Core (naive + AVX2 vector/matrix ops, benchmarks vs NumPy) | IN PROGRESS | — |
| 2 | Dense solvers (LU, Cholesky, QR, symmetric eigensolver) | PLANNED | 1 |
| 3 | CPU parallelism (Rayon) | PLANNED | 1, 2 |
| 4 | GPU backend (CUDA) | PLANNED | 1, f32 support; 2 for solver GEMM |
| 5 | Sparse (CSR, SpMV/SpMM, conjugate gradient) | PLANNED | 1; 4 for GPU SpMV |
| 6 | Fusion and small JIT | PLANNED | 4, 9e |
| 7 | Interop (C ABI, Python bindings) | PLANNED | a stable API from 1 (and later layers) |
| 8 | Qualification harness | IN PROGRESS | grows with every layer |
| 9 | Simulation workloads and neural network | PLANNED | see sub-items |

## Layer 1 — Core

**Goal.** Naive and AVX2 implementations of vector and matrix operations on `f64`, validated
against hand-calculated values and against each other, and benchmarked against NumPy across a
sweep of input sizes.

**Scope.**

- Vector operations: addition, subtraction, scaling, dot product, outer product, Euclidean (L2)
  and Manhattan (L1) norms.
- Matrix operations: addition, subtraction, scaling, matrix–matrix multiplication,
  matrix–vector multiplication, transpose, Frobenius norm, 1-norm, ∞-norm, and determinants for
  1×1, 2×2 and 3×3 only.
- Three implementations of each: naive Rust, AVX2 SIMD via `std::arch`, and a NumPy reference.
- Cross product for 3D vectors, as an early follow-up.

**Status by item.**

| Item | Status | Notes |
|---|---|---|
| Naive vector operations (7) and their golden-value tests | DONE | 24 tests in `naive/vector.rs`. |
| Naive matrix operations (12) and their golden-value tests | DONE | 35 tests in `naive/matrix.rs`. |
| AVX2 primitives (`load`, `store`, `add`) | IN PROGRESS | In `simd/primitives.rs`. |
| AVX2 vector operations | IN PROGRESS | `simd::vector::add` is started and unfinished; no other SIMD operation exists. |
| AVX2 matrix operations | PLANNED | |
| SIMD golden-value tests | PLANNED | No SIMD test exists yet. |
| Cross-tier differential tests (`proptest`) | PLANNED | Needs a second tier to compare against. |
| NumPy reference scripts | PLANNED | `scripts/` is empty. |
| `criterion` benchmarks and size sweep | PLANNED | `celeris/benches/` is empty and `criterion` is not yet a dependency. |
| Benchmark analysis tool (`celeris-analysis`) | PLANNED | The crate exists as an empty `main`. |
| Cross product (3D) | PLANNED | |
| Runtime CPU feature detection and per-instruction-set dispatch | PLANNED | The MVP hardcodes AVX2; see [DESIGN.md](../DESIGN.md). |

**Acceptance criteria.**

- Every operation has golden-value tests at each tier (naive, SIMD).
- The tiers agree with one another, and with NumPy, within the tolerance model in
  [VALIDATION.md](VALIDATION.md), on randomly generated inputs.
- Benchmark results are reported only once the benchmarks exist, with hardware metadata recorded.

**Out of scope.** Runtime-sized vectors and matrices (see
[Open design questions](#open-design-questions)), `f32` (pulled forward under layer 4), AVX-512
(research only), and a general N×N determinant (belongs to layer 2).

## Layer 2 — Dense solvers

**Goal.** Direct dense solvers built on Celeris's own matrix multiplication, validated against
LAPACK.

**Scope.** Blocked LU with partial pivoting, Cholesky, Householder QR, and a symmetric
eigensolver. A general N×N determinant (via LU) lives here too. Floating-point behavior and
conditioning need explicit attention: test matrices include ill-conditioned cases, not just
well-conditioned ones.

**Depends on.** Layer 1 (matrix multiplication as the GEMM building block), and an answer to the
storage-size question under [Open design questions](#open-design-questions).

**Acceptance criteria.** Residual and backward-error metrics (defined in
[VALIDATION.md](VALIDATION.md)) within thresholds chosen when the solvers are implemented, and
agreement with LAPACK (through SciPy) on the same inputs.

**Status.** PLANNED.

**Out of scope.** Banded and sparse factorizations, complex arithmetic, and any decomposition
not listed above (for example SVD or non-symmetric eigenproblems) until it is added here.

## Layer 3 — CPU parallelism

**Goal.** Multi-core scaling of layers 1 and 2 with Rayon, layered on top of SIMD.

**Scope.** Parallelized matrix multiplication and solver building blocks, with scaling
benchmarks across thread counts.

**Acceptance criteria.** Parallel results agree with serial results within tolerance; scaling
curves are published once measured.

**Status.** PLANNED.

**Out of scope.** MPI and multi-node execution.

## Layer 4 — GPU backend

**Goal.** A CUDA backend, built up through a matrix-multiplication progression, with an honest
comparison to cuBLAS.

**Scope.**

1. A naive CUDA matmul kernel.
2. A shared-memory tiled kernel.
3. Further tuning.
4. Each stage benchmarked, then compared against cuBLAS with an explanation of the remaining
   gap.
5. Moving the solvers' GEMM onto the GPU.

Work starts in `f32` (see the design notes below). The existing CPU/GPU crossover heuristic
(benchmark GPU against SIMD to find the input size where the GPU wins despite transfer cost)
belongs here.

**Depends on.** Layer 1 as the reference implementation, `f32` support, layer 8 for validation,
and layer 2 for the solver-GEMM step.

**Acceptance criteria.** GPU results match the CPU reference within an `f32`-appropriate
tolerance. Benchmarks against cuBLAS are reported honestly, including where Celeris loses.

**Status.** PLANNED.

**Out of scope.** Multi-GPU, non-NVIDIA backends (a vendor-neutral backend may follow later),
and multi-node execution.

**Design notes (not promises).**

- GPU kernels are written in CUDA C++, so this backend most likely means C++ kernels called from
  Rust, through the `cudarc` crate or direct FFI. The current state of that tooling, including
  NVRTC support, has **not been verified** — *to verify*.
- Consumer GeForce GPUs have heavily throttled `f64` throughput relative to `f32`, so an `f64`
  GPU benchmark would understate the implementation. `f32` support therefore needs to move ahead
  of the rest of the post-MVP list for this layer. The exact throttle ratio for the author's GPU
  is *to verify*.
- Hosted CI runners have no GPUs. CI can cover CPU backends only; GPU tests run locally or on a
  self-hosted runner.

## Layer 5 — Sparse

**Goal.** Sparse linear algebra primitives and an iterative solver.

**Scope.** CSR storage, SpMV and SpMM, conjugate gradient with a Jacobi preconditioner tested on
Poisson-style matrices, then GPU SpMV.

**Depends on.** Layer 1; the storage-size question (sparse formats are inherently
runtime-sized); layer 4 for GPU SpMV.

**Acceptance criteria.** SpMV and SpMM agree with SciPy's sparse routines; conjugate gradient
converges on Poisson-style systems with residuals checked against a reference solve.

**Status.** PLANNED.

**Out of scope.** Sparse formats other than CSR, direct sparse factorization, and
preconditioners other than Jacobi, until added here.

## Layer 6 — Fusion and small JIT

**Goal.** Runtime-compiled fused GPU kernels, for example matmul plus bias plus activation.

**Scope.** Kernel fusion through NVRTC, consumed by the neural network (9e).

**Depends on.** Layer 4, and 9e as its consumer. Whether NVRTC can be driven from Rust in
practice is *to verify*.

**Acceptance criteria.** A fused kernel's output matches the unfused composition within
tolerance, and the fused-versus-unfused benchmark is reported once it exists.

**Status.** PLANNED.

**Out of scope.** A general-purpose JIT or compiler framework.

## Layer 7 — Interop

**Goal.** Make the library usable outside Rust.

**Scope.** A stable C ABI, Python bindings (PyO3), and a versioned API. The bindings are planned
as separate crates, so the core library's build stays free of Python (consistent with
[DESIGN.md](../DESIGN.md)).

**Depends on.** A stable public API from layer 1, and ideally the later layers' APIs too.

**Acceptance criteria.** The C header has smoke tests, the Python bindings are tested against
NumPy results, and the versioning policy is documented.

**Status.** PLANNED.

**Out of scope.** Bindings for other languages.

## Layer 8 — Qualification harness

**Goal.** One tolerance-based test suite that runs across the scalar, SIMD, and GPU backends,
plus performance-regression tracking over time. Built continuously alongside every layer, not
added at the end.

**Status.** IN PROGRESS. What exists today is golden-value tests for the naive tier only. There
is no cross-backend suite, no CI, and no regression tracking yet.

**Detail.** See [VALIDATION.md](VALIDATION.md).

## Layer 9 — Simulation workloads and neural network

**Goal.** Exercise the library on realistic workloads and validate it against established
reference implementations. The direction is a domain-agnostic simulation core: state held as a
flat vector, with explicit and implicit ODE/DAE dynamics and Jacobian support.

**Status.** PLANNED. Split by dependency:

| Item | Scope | Depends on |
|---|---|---|
| 9a | Explicit integrators: RK4, then adaptive RK45 | Layer 1 only |
| 9b | Implicit/stiff integrator (backward Euler or BDF) | Layer 2 (Newton steps need linear solves) |
| 9c | Batched ensembles of integrations on the GPU | Layers 4 and 9a |
| 9d | A stencil PDE such as heat diffusion, CPU first, then GPU | Layer 1; layer 4 for GPU |
| 9e | A small neural network built on Celeris primitives (forward pass at minimum, backprop as a goal), benchmarked against an equivalent PyTorch network | Layers 1 and 4; layer 6 for fusion |
| 9f | A short, math-focused paper on the underlying algorithms | After the above |

**Acceptance criteria.**

- Integrators are checked against problems with analytic solutions, including empirical
  convergence order.
- The neural network's outputs agree with PyTorch within tolerance, and comparative benchmarks
  are reported honestly.

**Out of scope.** Domain-specific model libraries, and training infrastructure beyond what the
small network needs.

## Ordering and dependencies

An earlier plan put the explicit ODE integrators immediately after layer 1, then an implicit
integrator, then domain framing. The layered roadmap exposes a conflict: an implicit integrator
needs linear solves, so it depends on layer 2, while the explicit integrators need nothing
beyond layer 1. Splitting layer 9 by dependency (above) resolves it. The proposed order of
starting work is:

1. Finish layer 1, growing layer 8 alongside it.
2. 9a explicit RK4/RK45.
3. Layer 2 dense solvers.
4. 9b implicit integrator.
5. Layer 3 Rayon.
6. Layer 4 GPU: the matmul progression in `f32`, then solver GEMM on the GPU, then the cuBLAS
   comparison.
7. Layer 5 sparse.
8. Layer 6 fusion, then 9e the neural network with its PyTorch comparison, then 9f the paper.
9. Layer 7 interop, plus 9c and 9d on the GPU.

This order is a starting point and is expected to change.

## Planned workspace structure

To keep the project from becoming one monolithic crate, the planned structure is a Cargo
workspace with roughly one crate per layer, so each layer can ship and be validated
independently. Nothing below exists yet; names are provisional. See
[DESIGN.md](../DESIGN.md) for how this relates to the current two-crate workspace.

| Crate | Layer | Status |
|---|---|---|
| `celeris` | 1 (and the shared types) | exists |
| `celeris-analysis` | 1 (benchmark analysis tool) | exists, empty |
| `celeris-solvers` | 2 | PLANNED |
| `celeris-gpu` | 4 (including the CUDA kernels) | PLANNED |
| `celeris-sparse` | 5 | PLANNED |
| `celeris-fusion` | 6 | PLANNED |
| `celeris-ffi`, `celeris-py` | 7 | PLANNED |
| `celeris-qualify` | 8 | PLANNED |
| `celeris-sim` | 9a–9d | PLANNED |
| `celeris-nn` | 9e | PLANNED |

Layer 3 (Rayon) is likely a feature flag on `celeris` and `celeris-solvers` rather than a
crate; that is undecided.

## Open design questions

- **Storage model.** The core currently uses compile-time-sized, stack-allocated
  vectors and matrices (const generics), and runtime-sized types are an explicit non-goal. Dense
  solvers at useful sizes, GPU buffers, sparse formats, and neural-network batches all need
  runtime sizes or heap storage. Whether that means a separate dynamic type, a storage
  abstraction, or something else is undecided.
- **`f32` integration.** GPU work needs `f32` first. Whether it arrives as parallel `f32` types
  or a generic scalar type, and what an AVX2 `f32` path (eight lanes instead of four) implies
  for the existing primitives, is undecided.
- **Layer 3 as crate or feature flag.** Undecided (see above).

## To verify

These are recorded as unverified and are not stated as fact anywhere else in the docs:

- The `cudarc` versus direct-FFI route to CUDA C++ kernels, and its current maintenance state.
- NVRTC availability and usability from Rust.
- The `f64`-to-`f32` throughput ratio on the author's GPU.
- Rayon scaling behavior on the target workloads.
- Concrete numeric thresholds for solver residuals and backward error.

## Out of scope for Celeris overall

- MPI and multi-node execution.
- Quantum computing work.

# Celeris

A Rust numerical computing core. It currently implements vector and matrix operations, with a
SIMD-accelerated tier in development, and is being extended into a layered library covering
dense and sparse linear algebra, CPU and GPU backends, and a cross-backend validation harness.

## Status

Early development. Status tags below describe only what exists in the repository today.

- **DONE** — naive Rust vector and matrix operations, with golden-value tests (59 tests).
- **IN PROGRESS** — AVX2 SIMD implementation. The primitives and the first vector operation are
  started and unfinished, so the repository may not build at every commit while this is under
  way.
- **PLANNED** — NumPy reference implementations, `criterion` benchmarks and the comparison
  report, and everything in the roadmap below. No benchmark results exist yet, so none are
  claimed.

## Roadmap

The full roadmap, with scope, dependencies, and acceptance criteria for each layer, is in
[docs/ROADMAP.md](docs/ROADMAP.md). In summary:

| # | Layer | Status |
|---|---|---|
| 1 | Core: naive + AVX2 vector/matrix ops, benchmarks vs NumPy | IN PROGRESS |
| 2 | Dense solvers: LU, Cholesky, QR, symmetric eigensolver | PLANNED |
| 3 | CPU parallelism with Rayon | PLANNED |
| 4 | GPU backend: CUDA kernels, matmul progression, comparison against cuBLAS | PLANNED |
| 5 | Sparse: CSR, SpMV/SpMM, conjugate gradient | PLANNED |
| 6 | Fusion and small JIT: runtime-compiled fused kernels | PLANNED |
| 7 | Interop: C ABI and Python bindings | PLANNED |
| 8 | Qualification harness: cross-backend tests and regression tracking | IN PROGRESS |
| 9 | Simulation workloads (ODE integrators, PDE) and a small neural network | PLANNED |

How results are validated is described in [docs/VALIDATION.md](docs/VALIDATION.md).

## Requirements

- Rust (stable channel — no nightly features required)
- An x86_64 CPU with AVX2 support
- Python 3 + NumPy (for benchmark comparison only, not a runtime dependency of the core)

## Project Structure

Celeris is a Cargo workspace with two crates today — `celeris` (the numerical core library) and
`celeris-analysis` (kept separate so the core library's dependencies stay lean for downstream
consumers — see DESIGN.md). A structure with roughly one crate per roadmap layer is planned; see
DESIGN.md and docs/ROADMAP.md.

```
/celeris
  /src
    vector.rs, matrix.rs — the Vector<N> and Matrix<M, N> types
    /naive               — scalar reference implementations of each operation
    /simd                — AVX2 intrinsics via std::arch, wrapped for internal use
  /benches               — planned: criterion benchmarks (naive Rust vs. SIMD Rust vs. NumPy)
/celeris-analysis — planned: combines criterion + pyperf output into one comparison report
/scripts          — planned: NumPy reference implementations + pyperf benchmarking script
/docs             — roadmap and validation strategy
```

## Scope

The first milestone (layer 1) covers linear algebra basics across both vectors and matrices,
using `f64` only. See DESIGN.md for architecture and design decisions.

## Running locally

Setup instructions to be added as the project takes shape. The test suite runs with
`cargo test --workspace`.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option — the standard convention for Rust crates (Rust itself is dual-licensed the
same way), giving downstream users the flexibility to pick whichever license fits their project.

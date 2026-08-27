# Celeris

A Rust-based SIMD-accelerated numerical computing core, implementing vectorized linear algebra operations and benchmarking them against naive scalar implementations and NumPy.

## Status

Early development. Core vector and matrix operations are being built incrementally.

## Requirements

- Rust (stable channel — no nightly features required)
- An x86_64 CPU with AVX2 support
- Python 3 + NumPy (for benchmark comparison only, not a runtime dependency of the core)

## Project Structure

Celeris is a Cargo workspace with two crates — `celeris` (the numerical core library) and
`celeris-analysis` (kept separate so the core library's dependencies stay lean for downstream
consumers — see DESIGN.md).

```
/celeris
  /src
    /vector    — vector operations (add, scale, dot product, etc.)
    /matrix    — matrix operations (multiply, transpose, etc.)
    /simd      — AVX2 intrinsics via std::arch, wrapped for internal use
  /benches     — criterion benchmarks (naive Rust vs. SIMD Rust vs. NumPy)
/celeris-analysis — combines criterion + pyperf output into one comparison report (Rust)
/scripts          — NumPy reference implementations + pyperf benchmarking script (Python)
```

## Scope

MVP covers full linear algebra basics across both vectors and matrices, using `f64` only.
See DESIGN.md for architecture details and the post-MVP roadmap.

## Running locally

Setup instructions to be added as the project takes shape.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option — the standard convention for Rust crates (Rust itself is dual-licensed the
same way), giving downstream users the flexibility to pick whichever license fits their project.

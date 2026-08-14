# Celeris

A Rust-based SIMD-accelerated numerical computing core, implementing vectorized linear algebra operations and benchmarking them against naive scalar implementations and NumPy.

## Status

Early development. Core vector and matrix operations are being built incrementally.

## Requirements

- Rust (stable channel — no nightly features required)
- An x86_64 CPU with AVX2 support
- Python 3 + NumPy (for benchmark comparison only, not a runtime dependency of the core)

## Project Structure

```
/src
  /vector    — vector operations (add, scale, dot product, etc.)
  /matrix    — matrix operations (multiply, transpose, etc.)
  /simd      — AVX2 intrinsics via std::arch, wrapped for internal use
/benches     — criterion benchmarks (naive Rust vs. SIMD Rust vs. NumPy)
/scripts     — NumPy reference implementations used for benchmark comparison
```

## Scope

MVP covers full linear algebra basics across both vectors and matrices, using `f64` only.
See DESIGN.md for architecture details and the post-MVP roadmap.

## Running locally

Setup instructions to be added as the project takes shape.

## License

TBD

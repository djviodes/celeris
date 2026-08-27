# Celeris

A Rust-based SIMD-accelerated numerical computing core, implementing vectorized linear algebra operations and benchmarking them against naive scalar implementations and NumPy.

## Status

Early development. Core vector and matrix operations are being built incrementally.

## Requirements

- Rust (stable channel — no nightly features required)
- An x86_64 CPU with AVX2 support
- Python 3 + NumPy (for benchmark comparison only, not a runtime dependency of the core)

## Project Structure

Celeris is a Cargo workspace with (at least) two crates — the numerical core library, and a
separate analysis/benchmarking tool crate (kept separate so the core library's dependencies stay
lean for downstream consumers — see DESIGN.md). Exact crate names not yet decided.

```
/<core crate>
  /src
    /vector    — vector operations (add, scale, dot product, etc.)
    /matrix    — matrix operations (multiply, transpose, etc.)
    /simd      — AVX2 intrinsics via std::arch, wrapped for internal use
  /benches     — criterion benchmarks (naive Rust vs. SIMD Rust vs. NumPy)
/<analysis crate> — combines criterion + pyperf output into one comparison report (Rust)
/scripts          — NumPy reference implementations + pyperf benchmarking script (Python)
```

## Scope

MVP covers full linear algebra basics across both vectors and matrices, using `f64` only.
See DESIGN.md for architecture details and the post-MVP roadmap.

## Running locally

Setup instructions to be added as the project takes shape.

## License

TBD

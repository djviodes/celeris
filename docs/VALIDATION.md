# Celeris — Validation

How Celeris establishes that its results are correct, across every implementation and, as they
are added, every backend. This document describes both what exists today and what is planned;
each part carries a status tag (**DONE**, **IN PROGRESS**, **PLANNED**) following the same
honesty rule as [ROADMAP.md](ROADMAP.md): a tag describes only what exists in the repository
today. Architecture and design decisions live in [DESIGN.md](../DESIGN.md).

## Current coverage

| What | Status | Detail |
|---|---|---|
| Naive golden-value tests, vectors | DONE | 24 tests: construction (success and failure paths), and each operation at small, normal and large magnitudes. |
| Naive golden-value tests, matrices | DONE | 35 tests across all 12 matrix operations. |
| SIMD golden-value tests | PLANNED | No SIMD test exists yet. |
| Cross-tier differential tests | PLANNED | Waits on a second complete tier. |
| NumPy comparison | PLANNED | `scripts/` is empty. |
| Benchmarks | PLANNED | No benchmark exists yet, so there are no performance results. |
| CI | PLANNED | No CI configuration exists; checks currently run locally. |

## Principles

1. **Independent truth first.** Each backend gets its own golden-value tests, checked against
   hand-calculated values rather than against another backend. Comparing two implementations
   only proves they agree, not that either is right; a bug shared across backends is invisible
   to comparison alone.
2. **Comparison second.** Once more than one backend exists, differential testing checks that
   they agree with each other and with reference libraries.
3. **Performance last.** Benchmarks only follow once correctness is established at every tier.
4. **No exact float equality.** Results are compared with a tolerance (below), because
   floating-point arithmetic is not associative and a correct SIMD or GPU implementation may
   differ from a scalar one in the last few bits.

## Tolerance model

Comparisons use a combined absolute and relative tolerance. In the `approx` crate's
`assert_relative_eq!`, a pair of values `a` and `b` passes if either of these holds:

```
|a − b| ≤ epsilon                              (absolute)
|a − b| ≤ max_relative × max(|a|, |b|)         (relative)
```

The relative check alone breaks down near zero, where the allowed difference shrinks to nothing
and demands near-exact equality. The absolute `epsilon` is a fixed floor that covers that case.

Current convention in the golden-value tests: `epsilon = 1e-14`, and `max_relative = 1e-13` for
large-magnitude cases. Tests with small or moderate values rely on `epsilon` alone.

**Calibration.** A tolerance should be comfortably wider than real rounding noise and far
narrower than any plausible bug. One unit in the last place (ULP) at magnitude `m` is about
`m × f64::EPSILON` (`f64::EPSILON ≈ 2.22e-16`), and a handful of chained operations accumulate a
few ULPs. The important checks when picking a value are:

- A hand-computed expected value must itself be verified with a calculator or an independent
  script, because a mistake in the expected value looks identical to a bug in the code.
- A test whose tolerance is many orders of magnitude looser than the observed error can pass
  while hiding a real defect, so tolerances are reviewed against measured error.
- A test run with no explicit tolerance inherits the crate default, which equals
  `f64::EPSILON` and leaves almost no margin; it should be avoided.

Other element types and backends will need their own calibration. `f32` has a machine epsilon
of about `1.19e-7` rather than `f64`'s `2.22e-16`, and GPU results are expected to be compared against a
reference with an `f32`-appropriate tolerance rather than the `f64` values above. These
tolerances are **PLANNED** and not yet chosen.

## Reference implementations

| Reference | Used for | Status |
|---|---|---|
| NumPy | Vector and matrix operations; performance comparison in layer 1 | PLANNED |
| SciPy / LAPACK | Dense solvers in layer 2 (LU, Cholesky, QR, symmetric eigensolver) | PLANNED |
| SciPy sparse | SpMV, SpMM and conjugate gradient in layer 5 | PLANNED |
| cuBLAS | GPU matrix-multiplication correctness and performance comparison in layer 4 | PLANNED |
| PyTorch | Neural-network outputs and performance comparison in layer 9e | PLANNED |

NumPy is invoked from the Rust test suite as a subprocess rather than embedded through PyO3, so
Python stays out of the core library's build (see [DESIGN.md](../DESIGN.md)).

## Test tiers

- **Golden-value tests** — hand-calculated inputs and outputs, per backend. Naive: **DONE**.
  SIMD and GPU: **PLANNED**.
- **Differential tests** — randomly generated inputs, generated and shrunk with `proptest`,
  checked for agreement across backends and against the reference implementations. **PLANNED.**
- **Property tests for solvers** — checks that define correctness without a reference (the
  residual and orthogonality measures below). **PLANNED.**
- **Cross-backend harness** — one suite that runs the same cases through the scalar, SIMD and
  GPU backends. Planned as layer 8. **PLANNED.**

## Solver validation (layer 2)

Solvers are judged by how well they satisfy the defining equations, not only by matching
LAPACK's output digit for digit. With `u` the unit roundoff and `n` the matrix dimension, the
planned measures are:

- **LU with partial pivoting:** `‖PA − LU‖ / (n · u · ‖A‖)`.
- **Linear solve:** normalized backward error `‖Ax − b‖ / (‖A‖ · ‖x‖ + ‖b‖)`.
- **Cholesky:** `‖A − LLᵀ‖ / ‖A‖`.
- **QR:** `‖A − QR‖ / ‖A‖` and orthogonality `‖QᵀQ − I‖`.
- **Symmetric eigensolver:** `‖Av − λv‖ / ‖A‖` for each eigenpair, orthogonality of the
  eigenvectors, and agreement of the eigenvalues with LAPACK.

Test inputs deliberately include ill-conditioned matrices, since a solver that only passes on
well-conditioned inputs has not been tested where numerical behavior matters. Thresholds are
**not yet chosen**; they will be set from measured behavior when the solvers exist, expressed as
small multiples of `n · u` and justified in the tests.

## Regression tracking

Performance regressions are to be tracked over time on top of `criterion`, which remains the
measurement engine. Using Bencher for this is a candidate and is undecided. **PLANNED.**

## CI limits

Hosted CI runners have no GPUs. CI can therefore cover only the scalar and SIMD CPU backends.
GPU tests run locally or on a self-hosted runner, and results from those runs are not part of
automated CI until such a runner exists. SIMD tests additionally require a CI runner whose CPU
supports AVX2.

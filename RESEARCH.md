# Celeris — Research Backlog

A running list of topics that came up while scoping or building Celeris, worth digging into for
David's own understanding, but that weren't themselves blocking a specific decision at the time
they were noted. Not a task list — pull from this whenever there's appetite to go deeper on a
topic, in no particular order or urgency.

## From: vector/matrix sizing decision (2026-08-21)

- **Rust const generics — full capabilities and current limitations.** The sizing decision only
  required knowing that const generics allow a fixed size to be part of a type. Worth
  understanding more broadly: what const generic *expressions* are, what's still nightly-only vs.
  stabilized, and how the feature has evolved (min_const_generics and beyond).
- **How established Rust linear algebra crates solve storage genericity.** `nalgebra` in
  particular supports both fixed-size and dynamically-sized vectors/matrices under one API via a
  `Dim`/`Const`/`Dyn` abstraction plus separate storage traits (`ArrayStorage` vs. `VecStorage`).
  Reading that implementation directly (rather than a secondhand summary) is a good way to see a
  real, production answer to the same design question Celeris just made a narrower call on.
- **Stack vs. heap allocation mechanics, in more depth.** The sizing discussion only went as deep
  as "stack avoids an allocator call and a level of indirection." The fuller picture — how heap
  allocators actually work (free lists, fragmentation, why `malloc`/`free` cost what they cost),
  and how cache locality interacts with allocation strategy — is a foundational systems topic
  worth understanding beyond just this one decision.
- **How NumPy represents arrays internally.** NumPy is Celeris's external reference point for the
  whole project, so understanding its internal array representation (strides, dtype handling, how
  it manages memory for arbitrarily-shaped arrays) is useful background for comparison, even
  though it wasn't required to make the sizing call.

## From: correctness testing & floating-point tolerance decision (2026-08-22)

- **Catastrophic cancellation, in depth.** Came up while working out why relative-tolerance
  comparisons struggle near zero (subtracting two near-equal-magnitude floats cancels most
  leading digits, leaving a result dominated by prior rounding error). The combined
  absolute+relative tolerance decision was made without needing the full mechanics here — worth
  reading up on properly for its own sake, not just as a footnote to the tolerance decision.

## From: CPU feature detection decision (2026-08-22)

- **The real instruction-set hierarchy for `f64` SIMD work.** AVX (without the 2) already added
  256-bit floating-point vector instructions; AVX2 mainly extended 256-bit width to integer
  operations. So "AVX vs. AVX2 vs. AVX-512" isn't three evenly-spaced steps for this project's
  purely-`f64` workload the way the naming implies — worth understanding the actual differences
  precisely before building the post-MVP multi-instruction-set dispatch.

## From: naive golden-value test magnitude scoping (2026-09-05)

- **Unit-convention practice in particle/molecular simulation codes.** Came up while deciding how
  extreme a magnitude naive's golden-value tests should cover for the eventual physics-sim use
  case. Many simulation codes rescale into "reduced units" specifically to keep quantities near
  order-1 rather than working in raw SI magnitudes — if that's standard practice, it changes what
  "realistic" magnitude actually means for this library, rather than assuming raw physical
  quantities (e.g. femtometer-scale) are the real target.
- **`f64` precision behavior across its representable range.** Three related but distinct things
  worth understanding precisely: (1) the smallest magnitude `f64` can represent at all (~1e-308,
  far past anything tested so far), (2) how much relative precision it retains at a given
  magnitude, and (3) what happens when values of very different magnitudes are combined in one
  operation (the tiny one can vanish entirely) — this third one is the same mechanism as the
  already-noted catastrophic cancellation research above, not a separate topic.

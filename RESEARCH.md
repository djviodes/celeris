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

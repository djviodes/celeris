//! Celeris — SIMD-accelerated linear algebra core.
//!
//! See DESIGN.md at the workspace root for architecture and design decisions.

#[repr(C, align(32))]
pub struct Vector<const N: usize> {
    elements: [f64; N],
}
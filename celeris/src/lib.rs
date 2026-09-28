//! Celeris — SIMD-accelerated linear algebra core.
//!
//! See DESIGN.md at the workspace root for architecture and design decisions.

mod matrix;
pub mod naive;
pub mod simd;
mod vector;

pub use matrix::{Matrix, MatrixError};
pub use vector::{Vector, VectorError};

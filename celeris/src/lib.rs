//! Celeris — SIMD-accelerated linear algebra core.
//!
//! See DESIGN.md at the workspace root for architecture and design decisions.

mod matrix;
pub mod naive;
mod vector;

pub use matrix::{Matrix, MatrixError};
pub use vector::{Vector, VectorError};

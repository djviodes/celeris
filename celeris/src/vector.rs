use std::error::Error;
use std::fmt;
use std::ops::{Index, IndexMut};

#[repr(C, align(32))]
#[derive(Debug)]
pub struct Vector<const N: usize> {
    elements: [f64; N],
}

#[derive(Debug)]
pub enum VectorError {
    InvalidLength { received: usize, expected: usize },
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::InvalidLength {
                received: size_received,
                expected: size_expected,
            } => write!(
                f,
                "Size received {size_received} | Size expected {size_expected}"
            ),
        }
    }
}

impl Error for VectorError {}

impl<const N: usize> From<[f64; N]> for Vector<N> {
    fn from(source: [f64; N]) -> Self {
        Vector { elements: source }
    }
}

impl<const N: usize> TryFrom<&[f64]> for Vector<N> {
    type Error = VectorError;

    fn try_from(vector: &[f64]) -> Result<Self, Self::Error> {
        let elements: [f64; N] = vector.try_into().map_err(|_| VectorError::InvalidLength {
            received: vector.len(),
            expected: N,
        })?;
        Ok(Self { elements })
    }
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;

    #[track_caller]
    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl<const N: usize> Vector<N> {
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&f64> {
        (index < N).then(|| &self.elements[index])
    }

    #[must_use]
    pub fn len(&self) -> usize {
        N
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        N == 0
    }
}

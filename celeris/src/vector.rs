use std::error::Error;
use std::fmt;

#[repr(C, align(32))]
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

    fn try_from(vec: &[f64]) -> Result<Self, Self::Error> {
        let elements: [f64; N] = vec.try_into().map_err(|_| VectorError::InvalidLength {
            received: vec.len(),
            expected: N,
        })?;
        Ok(Self { elements })
    }
}

impl<const N: usize> Vector<N> {
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&f64> {
        (index < N).then(|| &self.elements[index])
    }
}

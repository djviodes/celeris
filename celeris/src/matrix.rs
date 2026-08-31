use std::error::Error;
use std::fmt;

#[repr(C, align(32))]
pub struct Matrix<const M: usize, const N: usize> {
    elements: [[f64; M]; N],
}

#[derive(Debug)]
pub enum MatrixError {
    InvalidColumnLength {
        columns_received: usize,
        columns_expected: usize,
    },
    InvalidRowLength {
        rows_received: usize,
        rows_expected: usize,
    },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::InvalidColumnLength {
                columns_received: column_size_received,
                columns_expected: columns_size_expected,
            } => write!(
                f,
                "Column size received {column_size_received} | Column size expected {columns_size_expected}"
            ),
            MatrixError::InvalidRowLength {
                rows_received: row_size_received,
                rows_expected: row_size_expected,
            } => write!(
                f,
                "Row size received {row_size_received} | Row size expected {row_size_expected}"
            ),
        }
    }
}

impl Error for MatrixError {}

impl<const M: usize, const N: usize> From<[[f64; M]; N]> for Matrix<M, N> {
    fn from(source: [[f64; M]; N]) -> Self {
        Matrix { elements: source }
    }
}

impl<const M: usize, const N: usize> TryFrom<&[&[f64]]> for Matrix<M, N> {
    type Error = MatrixError;

    fn try_from(matrix: &[&[f64]]) -> Result<Self, Self::Error> {
        let columns: [&[f64]; N] =
            matrix
                .try_into()
                .map_err(|_| MatrixError::InvalidColumnLength {
                    columns_received: matrix.len(),
                    columns_expected: N,
                })?;

        let mut elements: [[f64; M]; N] = [[0.0; M]; N];
        for (i, column) in columns.into_iter().enumerate() {
            let element: [f64; M] =
                column
                    .try_into()
                    .map_err(|_| MatrixError::InvalidRowLength {
                        rows_received: column.len(),
                        rows_expected: M,
                    })?;

            elements[i] = element;
        }
        Ok(Self { elements })
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<&f64> {
        (row < M && column < N).then(|| &self.elements[column][row])
    }
}

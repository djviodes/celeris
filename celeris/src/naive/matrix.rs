use crate::Matrix;
use crate::Vector;
use std::array::from_fn;

impl<const M: usize, const N: usize> Matrix<M, N> {
    /// # Panics
    ///
    /// `add` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn add(addend_1: &Matrix<M, N>, addend_2: &Matrix<M, N>) -> Matrix<M, N> {
        let matrix: [[f64; M]; N] = from_fn(|column| {
            from_fn(|row| {
                addend_1
                    .get(row, column)
                    .expect("index from from_fn is always in bounds")
                    + addend_2
                        .get(row, column)
                        .expect("index from from_fn is always in bounds")
            })
        });

        Matrix::from(matrix)
    }

    /// # Panics
    ///
    /// `subtract` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn subtract(minuend: &Matrix<M, N>, subtrahend: &Matrix<M, N>) -> Matrix<M, N> {
        let matrix: [[f64; M]; N] = from_fn(|column| {
            from_fn(|row| {
                minuend
                    .get(row, column)
                    .expect("index from from_fn is always in bounds")
                    - subtrahend
                        .get(row, column)
                        .expect("index from from_fn is always in bounds")
            })
        });

        Matrix::from(matrix)
    }

    /// # Panics
    ///
    /// `scale` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn scale(scalar: f64, matrix: &Matrix<M, N>) -> Matrix<M, N> {
        let scaled_matrix: [[f64; M]; N] = from_fn(|column| {
            from_fn(|row| {
                matrix
                    .get(row, column)
                    .expect("index from from_fn is always in bounds")
                    * scalar
            })
        });

        Matrix::from(scaled_matrix)
    }

    /// # Panics
    ///
    /// `multiply` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn multiply<const P: usize>(
        multiplicand: &Matrix<M, N>,
        multiplier: &Matrix<N, P>,
    ) -> Matrix<M, P> {
        let matrix: [[f64; M]; P] = from_fn(|p| {
            from_fn(|m| {
                (0..N)
                    .map(|n| {
                        multiplicand
                            .get(m, n)
                            .expect("index from from_fn is always in bounds")
                            * multiplier
                                .get(n, p)
                                .expect("index from from_fn is always in bounds")
                    })
                    .sum()
            })
        });

        Matrix::from(matrix)
    }

    /// # Panics
    ///
    /// `matrix_vector_multiplication` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn matrix_vector_multiplication(matrix: &Matrix<M, N>, vector: &Vector<N>) -> Vector<M> {
        let array: [f64; M] = from_fn(|m| {
            (0..N)
                .map(|n| {
                    matrix
                        .get(m, n)
                        .expect("index from from_fn is always in bounds")
                        * vector
                            .get(n)
                            .expect("index from from_fn is always in bounds")
                })
                .sum()
        });

        Vector::from(array)
    }
}

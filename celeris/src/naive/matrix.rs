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

    /// # Panics
    ///
    /// `transpose` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn transpose(matrix: &Matrix<M, N>) -> Matrix<N, M> {
        let transposed_matrix: [[f64; N]; M] = from_fn(|m| {
            from_fn(|n| {
                *matrix
                    .get(m, n)
                    .expect("index from from_fn is always in bounds")
            })
        });

        Matrix::from(transposed_matrix)
    }

    /// # Panics
    ///
    /// `frobenius_norm` cannot panic because the `map` iterator only iterates between 0 and
    /// N - 1 and then 0 and M - 1 elements which is always the size of the parameter vectors
    #[must_use]
    pub fn frobenius_norm(matrix: &Matrix<M, N>) -> f64 {
        let value: f64 = (0..N)
            .map(|n| {
                (0..M)
                    .map(|m| {
                        matrix
                            .get(m, n)
                            .expect("index from the map iterator is always in bounds")
                            .powi(2)
                    })
                    .sum::<f64>()
            })
            .sum();

        value.sqrt()
    }

    /// # Panics
    ///
    /// `one_norm` cannot panic because the `map` iterator only iterates between 0 and N - 1
    /// and then 0 and M - 1 elements which is always the size of the parameter vectors
    #[must_use]
    pub fn one_norm(matrix: &Matrix<M, N>) -> f64 {
        (0..N)
            .map(|n| {
                (0..M)
                    .map(|m| {
                        f64::abs(
                            *matrix
                                .get(m, n)
                                .expect("index from the map iterator is always in bounds"),
                        )
                    })
                    .sum()
            })
            .fold(0.0, f64::max)
    }

    /// # Panics
    ///
    /// `infinity_norm` cannot panic because the `map` iterator only iterates between 0 and N - 1
    /// and then 0 and M - 1 elements which is always the size of the parameter vectors
    #[must_use]
    pub fn infinity_norm(matrix: &Matrix<M, N>) -> f64 {
        (0..M)
            .map(|m| {
                (0..N)
                    .map(|n| {
                        f64::abs(
                            *matrix
                                .get(m, n)
                                .expect("index from the map iterator is always in bounds"),
                        )
                    })
                    .sum()
            })
            .fold(0.0, f64::max)
    }
}

impl Matrix<1, 1> {
    /// # Panics
    ///
    /// `determinant_one_by_one` cannot panic because a one by one matrix is guaranteed to have one element
    #[must_use]
    pub fn determinant_one_by_one(matrix: &Matrix<1, 1>) -> f64 {
        *matrix
            .get(0, 0)
            .expect("hardcoded index is always in bounds")
    }
}

impl Matrix<2, 2> {
    /// # Panics
    ///
    /// `determinant_two_by_two` cannot panic because a two by two matrix is guaranteed to have four elements
    #[must_use]
    pub fn determinant_two_by_two(matrix: &Matrix<2, 2>) -> f64 {
        let a: f64 = *matrix
            .get(0, 0)
            .expect("hardcoded index is always in bounds");

        let d: f64 = *matrix
            .get(1, 1)
            .expect("hardcoded index is always in bounds");

        let b: f64 = *matrix
            .get(0, 1)
            .expect("hardcoded index is always in bounds");

        let c: f64 = *matrix
            .get(1, 0)
            .expect("hardcoded index is always in bounds");

        a * d - b * c
    }
}

impl Matrix<3, 3> {
    /// # Panics
    ///
    /// `determinant_three_by_three` cannot panic because a three by three matrix is guaranteed to have nine elements
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn determinant_three_by_three(matrix: &Matrix<3, 3>) -> f64 {
        let a: f64 = *matrix
            .get(0, 0)
            .expect("hardcoded index is always in bounds");

        let b: f64 = *matrix
            .get(0, 1)
            .expect("hardcoded index is always in bounds");

        let c: f64 = *matrix
            .get(0, 2)
            .expect("hardcoded index is always in bounds");

        let d: f64 = *matrix
            .get(1, 0)
            .expect("hardcoded index is always in bounds");

        let e: f64 = *matrix
            .get(1, 1)
            .expect("hardcoded index is always in bounds");

        let f: f64 = *matrix
            .get(1, 2)
            .expect("hardcoded index is always in bounds");

        let g: f64 = *matrix
            .get(2, 0)
            .expect("hardcoded index is always in bounds");

        let h: f64 = *matrix
            .get(2, 1)
            .expect("hardcoded index is always in bounds");

        let i: f64 = *matrix
            .get(2, 2)
            .expect("hardcoded index is always in bounds");

        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }
}

#[cfg(test)]
mod tests {
    use crate::MatrixError;
    use approx::assert_relative_eq;
    use assert_matches::assert_matches;

    use super::*;

    #[test]
    fn initializing_normal_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(matrix.len(), 4);
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
    }

    #[test]
    fn initializing_illegal_rows_matrix_should_return_error() {
        let matrix: MatrixError =
            Matrix::<2, 2>::try_from(&[&[1.0][..], &[3.0, 4.0][..]][..]).unwrap_err();

        assert_matches!(
            matrix,
            MatrixError::InvalidRowLength {
                rows_received: 1,
                rows_expected: 2
            }
        );
    }

    #[test]
    fn initializing_illegal_columns_matrix_should_return_error() {
        let matrix: MatrixError = Matrix::<2, 2>::try_from(&[&[1.0, 2.0][..]][..]).unwrap_err();

        assert_matches!(
            matrix,
            MatrixError::InvalidColumnLength {
                columns_received: 1,
                columns_expected: 2
            }
        );
    }

    #[test]
    fn adding_normal_number_matrices_should_return_success() {
        let addend_1: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let addend_2: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);

        let summed_matrix: Matrix<2, 2> = Matrix::add(&addend_1, &addend_2);

        assert_relative_eq!(summed_matrix[(0, 0)], 2.0, epsilon = 1e-14);
        assert_relative_eq!(summed_matrix[(1, 0)], 4.0, epsilon = 1e-14);
        assert_relative_eq!(summed_matrix[(0, 1)], 6.0, epsilon = 1e-14);
        assert_relative_eq!(summed_matrix[(1, 1)], 8.0, epsilon = 1e-14);
    }

    #[test]
    fn adding_small_number_matrices_should_return_success() {
        let addend_1: Matrix<2, 2> = Matrix::from([[1e-12, 2e-12], [3e-12, 4e-12]]);
        let addend_2: Matrix<2, 2> = Matrix::from([[1e-12, 2e-12], [3e-12, 4e-12]]);

        let summed_matrix: Matrix<2, 2> = Matrix::add(&addend_1, &addend_2);

        assert_relative_eq!(summed_matrix[(0, 0)], 2e-12, epsilon = 1e-14);
        assert_relative_eq!(summed_matrix[(1, 0)], 4e-12, epsilon = 1e-14);
        assert_relative_eq!(summed_matrix[(0, 1)], 6e-12, epsilon = 1e-14);
        assert_relative_eq!(summed_matrix[(1, 1)], 8e-12, epsilon = 1e-14);
    }

    #[test]
    fn adding_large_number_matrices_should_return_success() {
        let addend_1: Matrix<2, 2> = Matrix::from([
            [1.000_000_000_000_001_3e14, 2.000_000_000_000_002_2e14],
            [3.000_000_000_000_004e14, 4.000_000_000_000_004_4e14],
        ]);
        let addend_2: Matrix<2, 2> = Matrix::from([
            [2.000_000_000_000_003e14, 4.000_000_000_000_001e14],
            [3.000_000_000_000_001e14, 1.000_000_000_000_007_5e14],
        ]);

        let summed_matrix: Matrix<2, 2> = Matrix::add(&addend_1, &addend_2);

        assert_relative_eq!(
            summed_matrix[(0, 0)],
            3.000_000_000_000_004_4e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            summed_matrix[(1, 0)],
            6.000_000_000_000_004e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            summed_matrix[(0, 1)],
            6.000_000_000_000_005e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            summed_matrix[(1, 1)],
            5.000_000_000_000_012e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }
}

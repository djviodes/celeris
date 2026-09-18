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

    #[test]
    fn subtracting_normal_number_matrices_should_return_success() {
        let minuend: Matrix<2, 2> = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
        let subtrahend: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);

        let difference_matrix: Matrix<2, 2> = Matrix::subtract(&minuend, &subtrahend);

        assert_relative_eq!(difference_matrix[(0, 0)], 4.0, epsilon = 1e-14);
        assert_relative_eq!(difference_matrix[(1, 0)], 4.0, epsilon = 1e-14);
        assert_relative_eq!(difference_matrix[(0, 1)], 4.0, epsilon = 1e-14);
        assert_relative_eq!(difference_matrix[(1, 1)], 4.0, epsilon = 1e-14);
    }

    #[test]
    fn subtracting_small_number_matrices_should_return_success() {
        let minuend: Matrix<2, 2> = Matrix::from([[5e-13, 6e-13], [7e-13, 8e-13]]);
        let subtrahend: Matrix<2, 2> = Matrix::from([[1e-13, 2e-13], [3e-13, 4e-13]]);

        let difference_matrix: Matrix<2, 2> = Matrix::subtract(&minuend, &subtrahend);

        assert_relative_eq!(difference_matrix[(0, 0)], 4e-13, epsilon = 1e-14);
        assert_relative_eq!(difference_matrix[(1, 0)], 4e-13, epsilon = 1e-14);
        assert_relative_eq!(difference_matrix[(0, 1)], 4e-13, epsilon = 1e-14);
        assert_relative_eq!(difference_matrix[(1, 1)], 4e-13, epsilon = 1e-14);
    }

    #[test]
    fn subtracting_large_number_matrices_should_return_success() {
        let minuend: Matrix<2, 2> = Matrix::from([
            [5.000_000_000_000_009e14, 6.000_000_000_000_002e14],
            [7.000_000_000_000_006e14, 8.000_000_000_000_009e14],
        ]);
        let subtrahend: Matrix<2, 2> = Matrix::from([
            [1.000_000_000_000_003e14, 2.000_000_000_000_001e14],
            [3.000_000_000_000_004e14, 4.000_000_000_000_003e14],
        ]);

        let difference_matrix: Matrix<2, 2> = Matrix::subtract(&minuend, &subtrahend);

        assert_relative_eq!(
            difference_matrix[(0, 0)],
            4.000_000_000_000_006e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            difference_matrix[(1, 0)],
            4.000_000_000_000_001e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            difference_matrix[(0, 1)],
            4.000_000_000_000_002e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            difference_matrix[(1, 1)],
            4.000_000_000_000_006e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }

    #[test]
    fn scaling_normal_number_matrices_should_return_success() {
        let scalar: f64 = 2.0;
        let matrix: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);

        let scaled_matrix: Matrix<2, 2> = Matrix::scale(scalar, &matrix);

        assert_relative_eq!(scaled_matrix[(0, 0)], 2.0, epsilon = 1e-14);
        assert_relative_eq!(scaled_matrix[(1, 0)], 4.0, epsilon = 1e-14);
        assert_relative_eq!(scaled_matrix[(0, 1)], 6.0, epsilon = 1e-14);
        assert_relative_eq!(scaled_matrix[(1, 1)], 8.0, epsilon = 1e-14);
    }

    #[test]
    fn scaling_small_number_matrices_should_return_success() {
        let scalar: f64 = 2.0;
        let matrix: Matrix<2, 2> = Matrix::from([[1e-13, 2e-13], [3e-13, 4e-13]]);

        let scaled_matrix: Matrix<2, 2> = Matrix::scale(scalar, &matrix);

        assert_relative_eq!(scaled_matrix[(0, 0)], 2e-13, epsilon = 1e-14);
        assert_relative_eq!(scaled_matrix[(1, 0)], 4e-13, epsilon = 1e-14);
        assert_relative_eq!(scaled_matrix[(0, 1)], 6e-13, epsilon = 1e-14);
        assert_relative_eq!(scaled_matrix[(1, 1)], 8e-13, epsilon = 1e-14);
    }

    #[test]
    fn scaling_large_number_matrices_should_return_success() {
        let scalar: f64 = 2.0;
        let matrix: Matrix<2, 2> = Matrix::from([
            [1.000_000_000_000_002e14, 2.000_000_000_000_005e14],
            [3.000_000_000_000_003e14, 3.500_000_000_000_007e14],
        ]);

        let scaled_matrix: Matrix<2, 2> = Matrix::scale(scalar, &matrix);

        assert_relative_eq!(
            scaled_matrix[(0, 0)],
            2.000_000_000_000_004e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            scaled_matrix[(1, 0)],
            4.000_000_000_000_01e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            scaled_matrix[(0, 1)],
            6.000_000_000_000_006e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            scaled_matrix[(1, 1)],
            7.000_000_000_000_014e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }

    #[test]
    fn multiply_normal_number_matrices_should_return_success() {
        let multiplicand: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let multiplier: Matrix<2, 2> = Matrix::from([[2.0, 2.0], [2.0, 2.0]]);

        let multiplied_matrix: Matrix<2, 2> = Matrix::multiply(&multiplicand, &multiplier);

        assert_relative_eq!(multiplied_matrix[(0, 0)], 8.0, epsilon = 1e-14);
        assert_relative_eq!(multiplied_matrix[(1, 0)], 12.0, epsilon = 1e-14);
        assert_relative_eq!(multiplied_matrix[(0, 1)], 8.0, epsilon = 1e-14);
        assert_relative_eq!(multiplied_matrix[(1, 1)], 12.0, epsilon = 1e-14);
    }

    #[test]
    fn multiply_small_number_matrices_should_return_success() {
        let multiplicand: Matrix<2, 2> = Matrix::from([[1e-13, 2e-13], [3e-13, 4e-13]]);
        let multiplier: Matrix<2, 2> = Matrix::from([[2.0, 2.0], [2.0, 2.0]]);

        let multiplied_matrix: Matrix<2, 2> = Matrix::multiply(&multiplicand, &multiplier);

        assert_relative_eq!(multiplied_matrix[(0, 0)], 8e-13, epsilon = 1e-14);
        assert_relative_eq!(multiplied_matrix[(1, 0)], 1.2e-12, epsilon = 1e-14);
        assert_relative_eq!(multiplied_matrix[(0, 1)], 8e-13, epsilon = 1e-14);
        assert_relative_eq!(multiplied_matrix[(1, 1)], 1.2e-12, epsilon = 1e-14);
    }

    #[test]
    fn multiply_large_number_matrices_should_return_success() {
        let multiplicand: Matrix<2, 2> = Matrix::from([
            [1.000_000_000_000_02e13, 2.000_000_000_000_05e13],
            [3.000_000_000_000_03e13, 3.500_000_000_000_07e13],
        ]);
        let multiplier: Matrix<2, 2> = Matrix::from([[2.0, 2.0], [2.0, 2.0]]);

        let multiplied_matrix: Matrix<2, 2> = Matrix::multiply(&multiplicand, &multiplier);

        assert_relative_eq!(
            multiplied_matrix[(0, 0)],
            8.000_000_000_000_1e13,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            multiplied_matrix[(1, 0)],
            1.100_000_000_000_024e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            multiplied_matrix[(0, 1)],
            8.000_000_000_000_1e13,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            multiplied_matrix[(1, 1)],
            1.100_000_000_000_024e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }

    #[test]
    fn multiply_normal_number_matrix_vector_should_return_success() {
        let matrix: Matrix<3, 2> = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let vector: Vector<2> = Vector::from([2.0, 2.0]);

        let multiplied_vector: Vector<3> = Matrix::matrix_vector_multiplication(&matrix, &vector);

        assert_relative_eq!(multiplied_vector[0], 10.0, epsilon = 1e-14);
        assert_relative_eq!(multiplied_vector[1], 14.0, epsilon = 1e-14);
        assert_relative_eq!(multiplied_vector[2], 18.0, epsilon = 1e-14);
    }

    #[test]
    fn multiply_small_number_matrix_vector_should_return_success() {
        let matrix: Matrix<3, 2> = Matrix::from([[1e-13, 2e-13, 3e-13], [4e-13, 5e-13, 6e-13]]);
        let vector: Vector<2> = Vector::from([2.0, 2.0]);

        let multiplied_vector: Vector<3> = Matrix::matrix_vector_multiplication(&matrix, &vector);

        assert_relative_eq!(multiplied_vector[0], 1e-12, epsilon = 1e-14);
        assert_relative_eq!(multiplied_vector[1], 1.4e-12, epsilon = 1e-14);
        assert_relative_eq!(multiplied_vector[2], 1.8e-12, epsilon = 1e-14);
    }

    #[test]
    fn multiply_large_number_matrix_vector_should_return_success() {
        let matrix: Matrix<3, 2> = Matrix::from([
            [
                1.000_000_000_000_01e13,
                2.000_000_000_000_04e13,
                3.000_000_000_000_03e13,
            ],
            [
                4.000_000_000_000_08e13,
                5.000_000_000_000_01e13,
                6.000_000_000_000_09e13,
            ],
        ]);
        let vector: Vector<2> = Vector::from([2.0, 2.0]);

        let multiplied_vector: Vector<3> = Matrix::matrix_vector_multiplication(&matrix, &vector);

        assert_relative_eq!(
            multiplied_vector[0],
            1.000_000_000_000_018e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            multiplied_vector[1],
            1.400_000_000_000_01e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            multiplied_vector[2],
            1.800_000_000_000_024e14,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }

    #[test]
    fn transposing_a_matrix_should_return_success() {
        let matrix: Matrix<3, 2> = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let transposed_matrix: Matrix<2, 3> = Matrix::transpose(&matrix);

        assert_relative_eq!(transposed_matrix[(0, 0)], 1.0);
        assert_relative_eq!(transposed_matrix[(0, 1)], 2.0);
        assert_relative_eq!(transposed_matrix[(0, 2)], 3.0);
        assert_relative_eq!(transposed_matrix[(1, 0)], 4.0);
        assert_relative_eq!(transposed_matrix[(1, 1)], 5.0);
        assert_relative_eq!(transposed_matrix[(1, 2)], 6.0);
    }

    #[test]
    fn frobenius_norming_a_normal_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);

        let frobenius_normed_value: f64 = Matrix::frobenius_norm(&matrix);

        assert_relative_eq!(frobenius_normed_value, 30.0_f64.sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn frobenius_norming_a_small_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([[1e-6, 2e-6], [3e-6, 4e-6]]);

        let frobenius_normed_value: f64 = Matrix::frobenius_norm(&matrix);

        assert_relative_eq!(frobenius_normed_value, 3e-11_f64.sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn frobenius_norming_a_large_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> =
            Matrix::from([[1.000_001e5, 2.000_006e5], [3.000_007e5, 4.000_002e5]]);

        let frobenius_normed_value: f64 = Matrix::frobenius_norm(&matrix);

        assert_relative_eq!(
            frobenius_normed_value,
            3.000_008_400_009e11_f64.sqrt(),
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }

    #[test]
    fn one_norming_a_normal_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([[1.0, -2.0], [0.0, 4.0]]);

        let one_normed_value: f64 = Matrix::one_norm(&matrix);

        assert_relative_eq!(one_normed_value, 4.0, epsilon = 1e-14);
    }

    #[test]
    fn one_norming_a_small_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([[1e-13, 2e-13], [1e-13, 0.0]]);

        let one_normed_value: f64 = Matrix::one_norm(&matrix);

        assert_relative_eq!(one_normed_value, 3e-13, epsilon = 1e-14);
    }

    #[test]
    fn one_norming_a_large_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([
            [1.000_000_000_000_07e13, 2.000_000_000_000_03e13],
            [3.000_000_000_000_04e13, 4.000_000_000_000_15e13],
        ]);

        let one_normed_value: f64 = Matrix::one_norm(&matrix);

        assert_relative_eq!(
            one_normed_value,
            7.000_000_000_000_19e13,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }

    #[test]
    fn infinity_norming_a_normal_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([[1.0, 2.0], [3.0, -4.0]]);

        let infinity_normed_value: f64 = Matrix::infinity_norm(&matrix);

        assert_relative_eq!(infinity_normed_value, 6.0, epsilon = 1e-14);
    }

    #[test]
    fn infinity_norming_a_small_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([[1e-13, 0.0], [6e-13, 8e-13]]);

        let infinity_normed_value: f64 = Matrix::infinity_norm(&matrix);

        assert_relative_eq!(infinity_normed_value, 8e-13, epsilon = 1e-14);
    }

    #[test]
    fn infinity_norming_a_large_number_matrix_should_return_success() {
        let matrix: Matrix<2, 2> = Matrix::from([
            [1.000_000_000_000_09e13, 2.000_000_000_000_02e13],
            [5.000_000_000_000_11e13, 1.000_000_000_000_04e13],
        ]);

        let infinity_normed_value: f64 = Matrix::infinity_norm(&matrix);

        assert_relative_eq!(
            infinity_normed_value,
            6.000_000_000_000_20e13,
            epsilon = 1e-14,
            max_relative = 1e-13
        );
    }
}

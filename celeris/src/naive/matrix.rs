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

use crate::Matrix;
use crate::Vector;
use std::array::from_fn;

impl<const N: usize> Vector<N> {
    /// # Panics
    ///
    /// `add` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn add(addend_1: &Vector<N>, addend_2: &Vector<N>) -> Vector<N> {
        let array: [f64; N] = from_fn(|i| {
            addend_1
                .get(i)
                .expect("index from from_fn is always in bounds")
                + addend_2
                    .get(i)
                    .expect("index from from_fn is always in bounds")
        });

        Vector::from(array)
    }

    /// # Panics
    ///
    /// `subtract` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn subtract(minuend: &Vector<N>, subtrahend: &Vector<N>) -> Vector<N> {
        let array: [f64; N] = from_fn(|i| {
            minuend
                .get(i)
                .expect("index from from_fn is always in bounds")
                - subtrahend
                    .get(i)
                    .expect("index from from_fn is always in bounds")
        });

        Vector::from(array)
    }

    /// # Panics
    ///
    /// `scale` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn scale(scalar: f64, vector: &Vector<N>) -> Vector<N> {
        let array: [f64; N] = from_fn(|i| {
            vector
                .get(i)
                .expect("index from from_fn is always in bounds")
                * scalar
        });

        Vector::from(array)
    }

    /// # Panics
    ///
    /// `dot` cannot panic because the `map` iterator only iterates between 0 and N - 1 elements
    /// which is always the size of the parameter vectors
    #[must_use]
    pub fn dot(vector_1: &Vector<N>, vector_2: &Vector<N>) -> f64 {
        (0..N)
            .map(|i| {
                vector_1
                    .get(i)
                    .expect("index from the map iterator is always in bounds")
                    * vector_2
                        .get(i)
                        .expect("index from the map iterator is always in bounds")
            })
            .sum()
    }

    /// # Panics
    ///
    /// `outer` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn outer<const M: usize>(
        column_vector: &Vector<M>,
        row_vector: &Vector<N>,
    ) -> Matrix<M, N> {
        let matrix: [[f64; M]; N] = from_fn(|column| {
            from_fn(|row| {
                column_vector
                    .get(row)
                    .expect("index from from_fn is always in bounds")
                    * row_vector
                        .get(column)
                        .expect("index from from_fn is always in bounds")
            })
        });

        Matrix::from(matrix)
    }

    #[must_use]
    pub fn euclidean_norm(vector: &Vector<N>) -> f64 {
        f64::sqrt(Vector::dot(vector, vector))
    }

    /// # Panics
    ///
    /// `manhattan_norm` cannot panic because the `map` iterator only iterates between 0 and
    /// N - 1 elements which is always the size of the parameter vectors
    #[must_use]
    pub fn manhattan_norm(vector: &Vector<N>) -> f64 {
        (0..N)
            .map(|i| {
                f64::abs(
                    *vector
                        .get(i)
                        .expect("index from the map iterator is always in bounds"),
                )
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use crate::VectorError;
    use approx::assert_relative_eq;
    use assert_matches::assert_matches;

    use super::*;

    #[test]
    fn successful_vector_try_from_initialization() -> Result<(), Box<dyn std::error::Error>> {
        let vector: Vector<5> = Vector::try_from(&[1.0, 2.0, 3.0, 4.0, 5.0][..])?;

        assert_eq!(vector.len(), 5);

        Ok(())
    }

    #[test]
    fn failed_vector_try_from_initialization() {
        let vector: VectorError = Vector::<5>::try_from(&[1.0, 2.0, 3.0][..]).unwrap_err();

        assert_matches!(
            vector,
            VectorError::InvalidLength {
                received: 3,
                expected: 5
            }
        );
    }

    #[test]
    fn adding_normal_number_vectors_should_return_success() {
        let addend_1: Vector<5> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        let addend_2: Vector<5> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]);

        let summed_vector: Vector<5> = Vector::add(&addend_1, &addend_2);

        assert_relative_eq!(
            *summed_vector
                .get(0)
                .expect("hardcoded index is always in bounds"),
            2.0,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            *summed_vector
                .get(1)
                .expect("hardcoded index is always in bounds"),
            4.0,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            *summed_vector
                .get(2)
                .expect("hardcoded index is always in bounds"),
            6.0,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            *summed_vector
                .get(3)
                .expect("hardcoded index is always in bounds"),
            8.0,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            *summed_vector
                .get(4)
                .expect("hardcoded index is always in bounds"),
            10.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn adding_small_number_vectors_should_return_success() {
        let addend_1: Vector<5> = Vector::from([1e-12, 2e-12, 3e-12, 4e-12, 5e-12]);
        let addend_2: Vector<5> = Vector::from([1e-12, 2e-12, 3e-12, 4e-12, 5e-12]);

        let summed_vector: Vector<5> = Vector::add(&addend_1, &addend_2);

        assert_relative_eq!(
            *summed_vector
                .get(0)
                .expect("hardcoded index is always in bounds"),
            2e-12,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            *summed_vector
                .get(1)
                .expect("hardcoded index is always in bounds"),
            4e-12,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            *summed_vector
                .get(2)
                .expect("hardcoded index is always in bounds"),
            6e-12,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            *summed_vector
                .get(3)
                .expect("hardcoded index is always in bounds"),
            8e-12,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            *summed_vector
                .get(4)
                .expect("hardcoded index is always in bounds"),
            1e-11,
            epsilon = 1e-12
        );
    }

    #[test]
    fn adding_large_number_vectors_should_return_success() {
        let addend_1: Vector<5> = Vector::from([
            1.000_000_000_000_001_3e14,
            2.000_000_000_000_002_2e14,
            3.000_000_000_000_004e14,
            4.000_000_000_000_004_4e14,
            5.000_000_000_000_005_6e14,
        ]);
        let addend_2: Vector<5> = Vector::from([
            1.000_000_000_000_001_3e14,
            2.000_000_000_000_002_2e14,
            3.000_000_000_000_004e14,
            4.000_000_000_000_004_4e14,
            5.000_000_000_000_005_6e14,
        ]);

        let summed_vector: Vector<5> = Vector::add(&addend_1, &addend_2);

        assert_relative_eq!(
            *summed_vector
                .get(0)
                .expect("hardcoded index is always in bounds"),
            2.000_000_000_000_002_5e14,
            epsilon = 2.0,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            *summed_vector
                .get(1)
                .expect("hardcoded index is always in bounds"),
            4.000_000_000_000_004_4e14,
            epsilon = 2.0,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            *summed_vector
                .get(2)
                .expect("hardcoded index is always in bounds"),
            6.000_000_000_000_008e14,
            epsilon = 2.0,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            *summed_vector
                .get(3)
                .expect("hardcoded index is always in bounds"),
            8.000_000_000_000_009e14,
            epsilon = 2.0,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            *summed_vector
                .get(4)
                .expect("hardcoded index is always in bounds"),
            1.000_000_000_000_001_1e15,
            epsilon = 2.0,
            max_relative = 1e-13
        );
    }
}

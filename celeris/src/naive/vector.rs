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
}

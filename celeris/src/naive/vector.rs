use crate::Vector;
use std::array::from_fn;

impl<const N: usize> Vector<N> {
    /// # Panics
    ///
    /// `add` cannot panic because `from_fn`'s contract guarantees the index is valid
    #[must_use]
    pub fn add(vec_1: &Vector<N>, vec_2: &Vector<N>) -> Vector<N> {
        let array: [f64; N] = from_fn(|i| {
            vec_1
                .get(i)
                .expect("index from from_fn is always in bounds")
                + vec_2
                    .get(i)
                    .expect("index from from_fn is always in bounds")
        });

        Vector::from(array)
    }
}

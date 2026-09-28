use crate::Vector;
use crate::simd::primitives;

#[must_use]
pub fn add<const N: usize>(addend_1: &Vector<N>, addend_2: &Vector<N>) -> Vector<N> {
    let mut result: Vector<N> = Vector::from([0.0; N]);

    let aligned_ptr_1: *const f64 = addend_1.as_ptr();
    let aligned_ptr_2: *const f64 = addend_2.as_ptr();
    let aligned_mut_ptr: *mut f64 = result.as_mut_ptr();
}
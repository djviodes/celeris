use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_load_pd, _mm256_store_pd};

/// # Safety
/// - `aligned_ptr` must be 32-byte aligned
/// - `aligned_ptr` must be valid for reads of 4 contiguous f64s
#[target_feature(enable = "avx2")]
pub(super) unsafe fn load_elements(aligned_ptr: *const f64) -> __m256d {
    // SAFETY: This call is safe because `_mm256_load_pd` requires a 32-byte-aligned pointer that
    // is valid for reads of 4 contiguous f64s, which is precisely what this function's own
    // safety contract requires the caller to uphold.
    unsafe { _mm256_load_pd(aligned_ptr) }
}

/// # Safety
/// - `aligned_ptr` must be 32-byte aligned
/// - `aligned_ptr` must be valid for writes of 4 contiguous f64s
#[target_feature(enable = "avx2")]
pub(super) unsafe fn store_elements(aligned_ptr: *mut f64, vector: __m256d) {
    // SAFETY: This call is safe because `_mm256_store_pd` requires a 32-byte-aligned pointer that
    // is valid for writes of 4 contiguous f64s, which is precisely what this function's own
    // safety contract requires the caller to uphold.
    unsafe { _mm256_store_pd(aligned_ptr, vector) }
}

/// # Safety
/// - The caller must ensure that the executing CPU supports AVX2. This crate makes no runtime
///   check and assumes it unconditionally per its MVP scope. 
#[target_feature(enable = "avx2")]
pub(super) unsafe fn add_elements(addend_1: __m256d, addend_2: __m256d) -> __m256d {
    _mm256_add_pd(addend_1, addend_2)
}
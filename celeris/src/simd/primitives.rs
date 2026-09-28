use std::arch::x86_64::{__m256d, _mm256_load_pd};

/// # Safety
/// - `aligned_ptr` must be 32-byte aligned
/// - `aligned_ptr` must be valid for reads of 4 contiguous f64s
#[target_feature(enable = "avx2")]
unsafe fn load_elements(aligned_ptr: *const f64) -> __m256d {
    // SAFETY: this call is safe because `_mm256_load_pd` requires a 32-byte-aligned pointer that
    // is valid for reads of 4 contiguous f64s, which is precisely what this function's own
    // safety contract requires the caller to uphold.
    unsafe { _mm256_load_pd(aligned_ptr) }
}

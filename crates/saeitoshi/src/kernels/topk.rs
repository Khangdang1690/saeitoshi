//! TopK selection over a length-d_sae vector of pre-activations.
//!
//! Scalar reference uses a maintained min-heap of size k (`O(n log k)`),
//! which is optimal for k <<  n. SIMD paths (M3) use radix-select.

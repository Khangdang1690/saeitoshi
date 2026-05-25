//! Sparse decoder: gather k rows of `W_dec`, multiply by their feature
//! values, accumulate into the dense reconstruction. Implementation lands
//! in M1 (scalar) and M3 (SIMD gather).

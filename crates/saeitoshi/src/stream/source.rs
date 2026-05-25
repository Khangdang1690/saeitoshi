//! `ActivationSource` trait + `.npy` / `.safetensors` implementations.

use crate::error::Result;

pub trait ActivationSource: Send {
    /// `(n_tokens, d_in)`.
    fn shape(&self) -> (usize, usize);

    /// Read `tile.len() / d_in` rows starting at `row_start` into `tile`.
    /// Returns the number of rows actually read.
    fn read_tile(&mut self, row_start: usize, tile: &mut [f32]) -> Result<usize>;
}

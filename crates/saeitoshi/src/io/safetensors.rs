//! safetensors helpers — we use the `safetensors` crate but route every read
//! through this module so loaders share dtype + shape validation logic.

use safetensors::{Dtype, SafeTensors};

use crate::error::{Result, SaeError};

/// Read a safetensors file fully into memory and return the parsed view + the
/// owning buffer (so callers can hold tensor views without lifetime tricks).
///
/// For huge SAEs (`d_sae` ≈ 4M, 40+ GB at FP32) this should switch to
/// mmap; for the SAE sizes we ship today, full-buffer read is simpler.
pub struct SafetensorsFile {
    buf: Vec<u8>,
}

impl SafetensorsFile {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let buf = std::fs::read(path).map_err(|source| SaeError::Io {
            path: Some(path.to_path_buf()),
            source,
        })?;
        // Validate header parses up front.
        let _ = SafeTensors::deserialize(&buf)?;
        Ok(Self { buf })
    }

    pub fn view(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.buf)?)
    }
}

/// Read a named f32 tensor into an owned `Box<[f32]>`, validating dtype.
pub fn read_f32_tensor(st: &SafeTensors<'_>, name: &str) -> Result<(Box<[f32]>, Vec<usize>)> {
    let view = st
        .tensor(name)
        .map_err(|_| SaeError::MissingTensor(name.to_string()))?;
    if view.dtype() != Dtype::F32 {
        return Err(SaeError::UnsupportedDtype(format!(
            "{name}: expected f32, got {:?}",
            view.dtype()
        )));
    }
    let bytes = view.data();
    if bytes.len() % 4 != 0 {
        return Err(SaeError::Other(format!(
            "{name}: byte length {} not divisible by 4",
            bytes.len()
        )));
    }
    let n = bytes.len() / 4;
    let mut buf = Vec::<f32>::with_capacity(n);
    // Use bytemuck for safe little-endian copy. safetensors stores LE f32.
    let f32_slice: &[f32] = bytemuck::cast_slice(bytes);
    buf.extend_from_slice(f32_slice);
    Ok((buf.into_boxed_slice(), view.shape().to_vec()))
}

/// Transpose `[rows, cols]` row-major → `[cols, rows]` row-major.
pub fn transpose(input: &[f32], rows: usize, cols: usize) -> Box<[f32]> {
    let mut out = vec![0.0f32; rows * cols].into_boxed_slice();
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = input[r * cols + c];
        }
    }
    out
}

//! Weight-storage dtype views and dispatch helpers.

pub mod f16;
pub mod f32;
pub mod i8;

use half::f16 as Half;

/// Owning weight-storage variant. The encoder kernel matches on this at the
/// outer loop and dispatches to the dtype-specialized kernel.
#[derive(Debug)]
pub enum WeightStorage {
    F32(Box<[f32]>),
    F16(Box<[Half]>),
    I8(I8Block),
}

#[derive(Debug)]
pub struct I8Block {
    pub qweight: Box<[i8]>,
    /// Per-column scales (encoder) or per-row scales (decoder).
    pub scales: Box<[f32]>,
    /// Symmetric quantization stores `None`; asymmetric stores zero points.
    pub zeros: Option<Box<[i8]>>,
}

impl WeightStorage {
    pub fn nbytes(&self) -> usize {
        match self {
            WeightStorage::F32(b) => b.len() * std::mem::size_of::<f32>(),
            WeightStorage::F16(b) => b.len() * std::mem::size_of::<Half>(),
            WeightStorage::I8(blk) => {
                blk.qweight.len()
                    + blk.scales.len() * std::mem::size_of::<f32>()
                    + blk.zeros.as_ref().map_or(0, |z| z.len())
            }
        }
    }
}

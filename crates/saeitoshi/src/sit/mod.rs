//! The `.sit` on-disk format: magic + version + JSON tensor manifest +
//! 64-byte-aligned tensor blobs. Designed for mmap-and-go.
//!
//! Layout:
//! ```text
//! [0..8)    Magic = b"SAEITOSI"
//! [8..12)   Format version (u32 LE)
//! [12..16)  Header length H (u32 LE)
//! [16..16+H)  JSON header (UTF-8)
//! [pad to 64-byte boundary]
//! [tensor section]   one blob per manifest entry, each 64-byte aligned
//! ```
//!
//! v0 emits FP32 only. The manifest reserves dtype + scale_tensor slots so
//! v0.1 quantized weights can land without a format break.

pub mod header;

use std::io::Write;
use std::path::Path;

use header::{SitHeader, SparsifierKind, TensorEntry};

use crate::config::WeightDtype;
use crate::error::{Result, SaeError};
use crate::sae::{DecoderWeights, EncoderWeights, Sae, WeightLayout};
use crate::sparsify::Sparsifier;

/// Magic bytes at the start of every `.sit` file.
pub const MAGIC: [u8; 8] = *b"SAEITOSI";

/// Current `.sit` format version.
pub const FORMAT_VERSION: u32 = 1;

const ALIGN: usize = 64;

fn pad_to(target: usize, current: usize) -> usize {
    (target - current % target) % target
}

/// Write an SAE to a `.sit` file. v0 emits FP32 only.
pub fn write(sae: &Sae, path: &Path) -> Result<()> {
    // Build tensor blob and manifest with relative offsets.
    let mut blob: Vec<u8> = Vec::new();
    let mut entries: Vec<TensorEntry> = Vec::new();

    let push = |entries: &mut Vec<TensorEntry>,
                blob: &mut Vec<u8>,
                name: &str,
                shape: Vec<usize>,
                data: &[u8]| {
        let pad = pad_to(ALIGN, blob.len());
        blob.extend(std::iter::repeat_n(0u8, pad));
        entries.push(TensorEntry {
            name: name.to_string(),
            dtype: "f32".to_string(),
            shape,
            offset: blob.len() as u64,
            nbytes: data.len() as u64,
            scale_tensor: None,
        });
        blob.extend_from_slice(data);
    };

    let cfg = sae.config().clone();
    let d_in = cfg.d_in;
    let d_sae = cfg.d_sae;

    // On-disk W_enc is always row-major `[d_sae, d_in]` — the canonical
    // form. If the in-memory layout is packed (perf-v2 tiled backend),
    // unpack into a temporary buffer first. This keeps `.sit` files
    // backend-agnostic so they can be loaded under any feature set.
    let w_enc_row_major: std::borrow::Cow<'_, [f32]> = match sae.encoder().layout {
        WeightLayout::RowMajor => std::borrow::Cow::Borrowed(&sae.encoder().w_enc),
        #[cfg(feature = "perf-v2")]
        WeightLayout::PackedPanels { m_r } => std::borrow::Cow::Owned(
            crate::kernels::gemm::pack::unpack_w_enc(&sae.encoder().w_enc, d_sae, d_in, m_r)
                .into_vec(),
        ),
        #[cfg(not(feature = "perf-v2"))]
        WeightLayout::PackedPanels { .. } => {
            unreachable!("PackedPanels layout is only constructible with the perf-v2 feature")
        }
    };

    push(
        &mut entries,
        &mut blob,
        "W_enc",
        vec![d_sae, d_in],
        bytemuck::cast_slice(&w_enc_row_major),
    );
    push(
        &mut entries,
        &mut blob,
        "b_enc",
        vec![d_sae],
        bytemuck::cast_slice(&sae.encoder().b_enc),
    );
    push(
        &mut entries,
        &mut blob,
        "W_dec",
        vec![d_sae, d_in],
        bytemuck::cast_slice(&sae.decoder().w_dec),
    );
    push(
        &mut entries,
        &mut blob,
        "b_dec",
        vec![d_in],
        bytemuck::cast_slice(&sae.decoder().b_dec),
    );

    let kind = match sae.sparsifier() {
        Sparsifier::TopK {
            k,
            post_relu,
            rescale,
        } => {
            if let Some(r) = rescale {
                push(
                    &mut entries,
                    &mut blob,
                    "rescale",
                    vec![d_sae],
                    bytemuck::cast_slice(r),
                );
            }
            SparsifierKind::Topk {
                k: *k,
                post_relu: *post_relu,
                has_rescale: rescale.is_some(),
            }
        }
        Sparsifier::JumpReLU { thresholds } => {
            push(
                &mut entries,
                &mut blob,
                "threshold",
                vec![d_sae],
                bytemuck::cast_slice(thresholds),
            );
            SparsifierKind::Jumprelu
        }
        Sparsifier::Relu => SparsifierKind::Relu,
    };

    let header = SitHeader {
        config: cfg,
        sparsifier: kind,
        tensors: entries,
    };
    let json = serde_json::to_vec(&header)?;
    let header_len = json.len() as u32;

    let mut file = std::fs::File::create(path).map_err(|source| SaeError::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    file.write_all(&MAGIC)?;
    file.write_all(&FORMAT_VERSION.to_le_bytes())?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(&json)?;
    // Pad to 64-byte boundary.
    let pos = 16 + json.len();
    let pad = pad_to(ALIGN, pos);
    if pad > 0 {
        file.write_all(&vec![0u8; pad])?;
    }
    file.write_all(&blob)?;
    file.flush()?;
    Ok(())
}

/// Read an SAE from a `.sit` file.
pub fn read(path: &Path) -> Result<Sae> {
    let bytes = std::fs::read(path).map_err(|source| SaeError::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    if bytes.len() < 16 {
        return Err(SaeError::invalid(path, "file too small for sit header"));
    }
    if bytes[0..8] != MAGIC {
        return Err(SaeError::invalid(path, "missing SAEITOSI magic"));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(SaeError::invalid(
            path,
            format!("unsupported sit version {version} (expected {FORMAT_VERSION})"),
        ));
    }
    let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if bytes.len() < 16 + header_len {
        return Err(SaeError::invalid(path, "header_len exceeds file size"));
    }
    let header_json = &bytes[16..16 + header_len];
    let header: SitHeader = serde_json::from_slice(header_json)?;

    if header.config.weight_dtype != WeightDtype::F32 {
        return Err(SaeError::UnsupportedDtype(format!(
            "v0 sit reader only handles f32, got {:?}",
            header.config.weight_dtype
        )));
    }

    // Tensor section begins at the next 64-byte boundary after the header.
    let pos = 16 + header_len;
    let pad = pad_to(ALIGN, pos);
    let tensor_section_start = pos + pad;

    let read_tensor = |name: &str| -> Result<Box<[f32]>> {
        let entry = header
            .tensors
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| SaeError::MissingTensor(name.to_string()))?;
        if entry.dtype != "f32" {
            return Err(SaeError::UnsupportedDtype(entry.dtype.clone()));
        }
        let start = tensor_section_start + entry.offset as usize;
        let end = start + entry.nbytes as usize;
        if end > bytes.len() {
            return Err(SaeError::invalid(
                path,
                format!("tensor {name} extends past EOF"),
            ));
        }
        let f32_slice: &[f32] = bytemuck::cast_slice(&bytes[start..end]);
        Ok(f32_slice.to_vec().into_boxed_slice())
    };

    let d_in = header.config.d_in;
    let d_sae = header.config.d_sae;

    let w_enc = read_tensor("W_enc")?;
    let b_enc = read_tensor("b_enc")?;
    let w_dec = read_tensor("W_dec")?;
    let b_dec = read_tensor("b_dec")?;

    let sparsifier = match header.sparsifier {
        SparsifierKind::Topk {
            k,
            post_relu,
            has_rescale,
        } => {
            let rescale = if has_rescale {
                Some(read_tensor("rescale")?.into_vec())
            } else {
                None
            };
            Sparsifier::TopK {
                k,
                post_relu,
                rescale,
            }
        }
        SparsifierKind::Jumprelu => {
            let thresholds = read_tensor("threshold")?.into_vec();
            Sparsifier::JumpReLU { thresholds }
        }
        SparsifierKind::Relu => Sparsifier::Relu,
    };

    let enc = EncoderWeights {
        w_enc,
        b_enc,
        d_in,
        d_sae,
        layout: WeightLayout::RowMajor,
    };
    let dec = DecoderWeights {
        w_dec,
        b_dec,
        d_in,
        d_sae,
    };

    Sae::from_parts(header.config, enc, dec, sparsifier)
}

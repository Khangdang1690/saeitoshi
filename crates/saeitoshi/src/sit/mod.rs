//! The `.sit` on-disk format: magic + version + JSON tensor manifest +
//! 64-byte-aligned tensor blobs. Designed for mmap-and-go.
//!
//! Writer lands in M4 (FP32 only for v0); reader lands in M1/M2.

pub mod header;

/// Magic bytes at the start of every `.sit` file.
pub const MAGIC: [u8; 8] = *b"SAEITOSI";

/// Current `.sit` format version.
pub const FORMAT_VERSION: u32 = 1;

//! aarch64 NEON kernels.
//!
//! Reachable only via the `#[cfg(target_arch = "aarch64")]` arm in
//! [`crate::kernels::select_backend`]. Implementation lands in M3.

#![allow(unsafe_code)]

use super::Backend;

pub static NEON: Backend = Backend { name: "neon" };

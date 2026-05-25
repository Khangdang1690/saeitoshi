//! Python bindings for saeitoshi.
//!
//! This is a thin wrapper crate: every public function should release the GIL
//! around any saeitoshi call that does real work, and should never hold a
//! borrow into Python-owned memory across an FFI call back into Python.

use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Classes and functions are registered in M2.
    Ok(())
}

//! Python bindings for saeitoshi.
//!
//! Thin wrapper: every method releases the GIL around real work and never
//! holds a borrow into Python-owned memory across an FFI call back into
//! Python. Numeric I/O goes through numpy arrays.

// PyO3 0.22's `#[pymethods]` macro expands to code clippy flags as a
// `PyErr -> PyErr` useless conversion. Crate-wide allow rather than per-method.
#![allow(clippy::useless_conversion)]

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use saeitoshi::error::SaeError;
use saeitoshi::{Sae as RustSae, SparseOut as RustSparseOut};

fn map_err(e: SaeError) -> PyErr {
    match e {
        SaeError::Io { .. } => PyIOError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// Sparse feature output. CSR-style: `indices` and `values` are 1D arrays
/// of length `nnz`; `row_offsets[i]..row_offsets[i+1]` slices both for the
/// i-th batch row.
#[pyclass(name = "SparseFeatures", module = "sae._native")]
struct PySparseFeatures {
    inner: RustSparseOut,
}

#[pymethods]
impl PySparseFeatures {
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        let v: Vec<i32> = self.inner.indices.iter().map(|&x| x as i32).collect();
        v.into_pyarray_bound(py)
    }

    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.values.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn row_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        let v: Vec<i32> = self.inner.row_offsets.iter().map(|&x| x as i32).collect();
        v.into_pyarray_bound(py)
    }

    #[getter]
    fn d_sae(&self) -> u32 {
        self.inner.d_sae
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    fn __len__(&self) -> usize {
        self.inner.batch_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseFeatures(batch_size={}, nnz={}, d_sae={})",
            self.inner.batch_size(),
            self.inner.indices.len(),
            self.inner.d_sae,
        )
    }

    /// Materialize as a dense `[batch, d_sae]` numpy array. Expensive for
    /// large d_sae — prefer working with the CSR arrays directly.
    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let batch = self.inner.batch_size();
        let d_sae = self.inner.d_sae as usize;
        let dense = py.allow_threads(|| {
            let mut dense = vec![0.0f32; batch * d_sae];
            for b in 0..batch {
                let s = self.inner.row_offsets[b] as usize;
                let e = self.inner.row_offsets[b + 1] as usize;
                for k in s..e {
                    let idx = self.inner.indices[k] as usize;
                    dense[b * d_sae + idx] = self.inner.values[k];
                }
            }
            dense
        });
        let arr = Array2::from_shape_vec((batch, d_sae), dense)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }
}

/// Loaded SAE ready for inference. The Python-facing `sae.SAE` class
/// composes (does not subclass) this type so streaming and top-activations
/// helpers can be added in pure Python while leaving Rust pyclass attrs
/// simple.
#[pyclass(name = "NativeSAE", module = "sae._native")]
struct PySae {
    inner: RustSae,
}

#[pymethods]
impl PySae {
    /// Load from disk, auto-detecting the format. Currently supports a
    /// SAELens directory (`cfg.json` + `sae_weights.safetensors`).
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let sae = RustSae::load(path).map_err(map_err)?;
        Ok(Self { inner: sae })
    }

    #[getter]
    fn d_in(&self) -> usize {
        self.inner.d_in()
    }

    #[getter]
    fn d_sae(&self) -> usize {
        self.inner.d_sae()
    }

    #[getter]
    fn architecture(&self) -> String {
        format!("{:?}", self.inner.config().architecture).to_lowercase()
    }

    fn __repr__(&self) -> String {
        format!(
            "SAE(architecture={}, d_in={}, d_sae={})",
            self.architecture(),
            self.inner.d_in(),
            self.inner.d_sae()
        )
    }

    /// Encode `x: float32[B, d_in]` into a [`SparseFeatures`].
    fn encode(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f32>) -> PyResult<PySparseFeatures> {
        let arr = x.as_array();
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != self.inner.d_in() {
            return Err(PyValueError::new_err(format!(
                "expected shape [B, {}], got {:?}",
                self.inner.d_in(),
                shape
            )));
        }
        let batch = shape[0];
        let x_vec: Vec<f32> = arr.iter().copied().collect();
        let mut out = RustSparseOut::new(self.inner.d_sae() as u32);
        py.allow_threads(|| self.inner.encode(&x_vec, batch, &mut out))
            .map_err(map_err)?;
        Ok(PySparseFeatures { inner: out })
    }

    /// Decode a [`SparseFeatures`] back into the activation space.
    /// Returns `float32[batch_size, d_in]`.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        features: &PySparseFeatures,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let batch = features.inner.batch_size();
        let d_in = self.inner.d_in();
        let mut out = vec![0.0f32; batch * d_in];
        py.allow_threads(|| self.inner.decode(&features.inner, &mut out))
            .map_err(map_err)?;
        let arr = Array2::from_shape_vec((batch, d_in), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    /// Encode + decode in one shot. Returns `float32[B, d_in]`.
    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = x.as_array();
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != self.inner.d_in() {
            return Err(PyValueError::new_err(format!(
                "expected shape [B, {}], got {:?}",
                self.inner.d_in(),
                shape
            )));
        }
        let batch = shape[0];
        let d_in = self.inner.d_in();
        let x_vec: Vec<f32> = arr.iter().copied().collect();
        let mut out = vec![0.0f32; batch * d_in];
        py.allow_threads(|| self.inner.reconstruct(&x_vec, batch, &mut out))
            .map_err(map_err)?;
        let arr = Array2::from_shape_vec((batch, d_in), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PySae>()?;
    m.add_class::<PySparseFeatures>()?;
    Ok(())
}

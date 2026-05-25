"""saeitoshi — fast inference for Sparse Autoencoders.

PyPI name: ``saeitoshi``. Import name: ``sae``.

Quickstart::

    import sae
    model = sae.SAE.load("./my_sae")           # SAELens directory layout
    features = model.encode(activations)        # numpy float32[B, d_in]
    recon = model.decode(features)              # numpy float32[B, d_in]
"""

from __future__ import annotations

from ._native import SAE, SparseFeatures, __version__

__all__ = ["SAE", "SparseFeatures", "__version__"]

"""saeitoshi — fast inference for Sparse Autoencoders.

PyPI name: ``saeitoshi``. Import name: ``sae``.

Quickstart:

    >>> import sae
    >>> model = sae.SAE.load("./my_sae")           # SAELens directory layout
    >>> features = model.encode(activations)        # numpy float32[B, d_in]
    >>> recon = model.decode(features)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Native extension is wired up in M2.
# from . import _native  # noqa: F401

__all__ = ["__version__"]

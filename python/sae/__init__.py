"""saeitoshi — fast inference for Sparse Autoencoders.

PyPI name: ``saeitoshi``. Import name: ``sae``.

Quickstart::

    import sae
    model = sae.SAE.load("./my_sae")
    features = model.encode(activations)        # numpy float32[B, d_in]
    recon = model.decode(features)              # numpy float32[B, d_in]

    # Streaming over a large .npy file
    for batch in model.encode_stream("activations.npy", batch_size=8192):
        ...

    # Top-activating examples per feature
    top = model.top_activations("activations.npy", feature_ids=[1234, 5678], top_k=20)
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ._native import NativeSAE as _NativeSAE
from ._native import SparseFeatures, __version__

__all__ = ["SAE", "SparseFeatures", "__version__"]


def _coerce_activation_source(source) -> np.ndarray:
    """Normalize an activation source into a 2D float32-castable ndarray."""
    if isinstance(source, (str, Path)):
        arr = np.load(str(source), mmap_mode="r")
    elif isinstance(source, np.ndarray):
        arr = source
    else:
        raise TypeError(
            f"activation_source must be a path or ndarray, got {type(source).__name__}"
        )
    if arr.ndim != 2:
        raise ValueError(f"activation_source must be 2D [N, d_in], got shape {arr.shape}")
    return arr


class SAE:
    """A loaded Sparse Autoencoder.

    Thin Python wrapper over the Rust extension. Methods forward to the
    native object, with extra ``encode_stream`` / ``top_activations``
    helpers implemented in Python on top of the streaming API.
    """

    __slots__ = ("_native",)

    def __init__(self, native: _NativeSAE):
        self._native = native

    @classmethod
    def load(cls, path: str | Path) -> "SAE":
        return cls(_NativeSAE.load(str(path)))

    # ---- Properties forwarded to the native type ----

    @property
    def d_in(self) -> int:
        return self._native.d_in

    @property
    def d_sae(self) -> int:
        return self._native.d_sae

    @property
    def architecture(self) -> str:
        return self._native.architecture

    def __repr__(self) -> str:
        return repr(self._native)

    # ---- Hot path: encode / decode / reconstruct ----

    def encode(self, x: np.ndarray) -> SparseFeatures:
        return self._native.encode(x)

    def decode(self, features: SparseFeatures) -> np.ndarray:
        return self._native.decode(features)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self._native.reconstruct(x)

    # ---- Streaming + analysis utilities ----

    def encode_stream(self, source, *, batch_size: int = 8192) -> Iterator[SparseFeatures]:
        """Encode an activation source one tile at a time.

        Yields a :class:`SparseFeatures` per batch. The source is read
        memory-mapped when it's a ``.npy`` path, so peak memory is
        bounded by ``batch_size * d_in * 4`` plus the sparse output.
        """
        arr = _coerce_activation_source(source)
        n = arr.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = np.ascontiguousarray(arr[start:end], dtype=np.float32)
            yield self._native.encode(chunk)

    def top_activations(
        self,
        source,
        *,
        feature_ids: Sequence[int],
        top_k: int = 20,
        batch_size: int = 8192,
    ) -> Mapping[int, list[tuple[float, int]]]:
        """Find the top-k activating positions in `source` for each feature.

        Returns ``{feature_id: [(value, token_index), ...]}`` sorted
        descending by value. Token index is the global row index in
        `source`. Single pass over the activations.
        """
        fids = [int(f) for f in feature_ids]
        if not fids:
            return {}
        fid_arr = np.asarray(fids, dtype=np.int32)
        heaps: dict[int, list[tuple[float, int]]] = {f: [] for f in fids}

        base = 0
        for batch in self.encode_stream(source, batch_size=batch_size):
            indices = batch.indices
            values = batch.values
            row_offsets = batch.row_offsets
            if indices.size > 0:
                mask = np.isin(indices, fid_arr)
                if mask.any():
                    rel_positions = np.flatnonzero(mask)
                    # row_offsets[1:] is the cumulative nnz at the END of each row,
                    # so searchsorted(side='right') maps a position to its row.
                    row_of = np.searchsorted(row_offsets[1:], rel_positions, side="right")
                    masked_fids = indices[mask]
                    masked_vals = values[mask]
                    for fid, val, row in zip(masked_fids, masked_vals, row_of):
                        fid_i = int(fid)
                        val_f = float(val)
                        tok = int(base + row)
                        h = heaps[fid_i]
                        if len(h) < top_k:
                            heapq.heappush(h, (val_f, tok))
                        elif val_f > h[0][0]:
                            heapq.heapreplace(h, (val_f, tok))
            base += batch.batch_size

        return {f: sorted(heaps[f], reverse=True) for f in fids}

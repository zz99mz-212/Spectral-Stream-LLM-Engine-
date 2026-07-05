from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SpectralGeometry:
    """C6. SPECTRAL-GEOMETRY: Laplace-Beltrami eigenmaps + heat kernel."""

    name = "spectral_geometry"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_eig: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(n_eig, m, n)

        sim = t @ t.T
        sim = (sim + sim.T) * 0.5
        d = np.sum(np.abs(sim), axis=1) + 1e-10
        L = np.diag(d) - sim

        eigvals, eigvecs = np.linalg.eigh(L)
        _idx = np.argsort(eigvals)
        eigvals = eigvals[_idx]
        eigvecs = eigvecs[:, _idx]
        k = min(k + 1, len(eigvals))

        phi = eigvecs[:, 1:k]
        lam = eigvals[1:k]

        coeffs = phi.T @ t

        meta = dict(shape=t.shape, k=k - 1)
        data = (
            _serialize(phi.astype(np.float32))
            + _serialize(lam.astype(np.float32))
            + _serialize(coeffs.astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        m, n = shape

        phi = _deserialize(data[: m * k * 4]).reshape(m, k)
        pos = m * k * 4
        lam = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        coeffs = _deserialize(data[pos : pos + k * n * 4]).reshape(k, n)

        recon = phi @ coeffs
        return recon.astype(np.float32)

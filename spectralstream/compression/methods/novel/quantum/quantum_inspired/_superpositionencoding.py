from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SuperpositionEncoding:
    """Encode multiple weight values as quantum superposition amplitudes.
    W = Σ α_i |ψ_i⟩. Store only the dominant α_i coefficients after projecting
    onto a computational basis of size d.
    """

    name = "superposition_encoding"
    category = "quantum_compression"

    def __init__(self) -> None:
        self._rng = np.random.RandomState(42)

    def _build_basis(self, d: int, n: int) -> np.ndarray:
        basis = self._rng.randn(d, n).astype(np.float64)
        basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-30
        return basis

    def compress(
        self, tensor: np.ndarray, n_amplitudes: int = 16, basis_dim: int = 64
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        d = min(basis_dim, n)
        k = min(n_amplitudes, d)
        basis = self._build_basis(d, n)
        proj = basis @ t
        idx = np.argsort(-np.abs(proj))[:k]
        alphas = proj[idx].astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            d=d,
            k=k,
            indices=idx.astype(np.int32).tobytes(),
            seed=42,
        )
        data = struct.pack("<III", n, d, k)
        data += idx.astype(np.int32).tobytes()
        data += _serialize(alphas)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, d, k = struct.unpack_from("<III", data, 0)
        pos = 12
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32)
        pos += k * 4
        alphas = _deserialize(data[pos : pos + k * 4])
        self._rng.seed(metadata.get("seed", 42))
        basis = self._build_basis(d, n)
        recon = np.zeros(n, dtype=np.float64)
        for i in range(k):
            recon += alphas[i] * basis[idx[i]]
        return recon.reshape(shape).astype(np.float32)

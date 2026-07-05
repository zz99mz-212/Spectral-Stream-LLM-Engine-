from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumPCA:
    """Quantum PCA: density matrix exponentiation + phase estimation for
    eigenvectors. Simulate via iterative power method on the covariance.
    """

    name = "quantum_pca"
    category = "quantum_compression"

    def compress(
        self, tensor: np.ndarray, n_components: int = 8, n_power_iters: int = 10
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        m, n = t.shape
        d = min(n_components, m, n)
        cov = t.T @ t / m
        rng = np.random.RandomState(42)
        Q = rng.randn(n, d).astype(np.float64)
        Q, _ = np.linalg.qr(Q)
        for _ in range(n_power_iters):
            Q_new = cov @ Q
            Q, _ = np.linalg.qr(Q_new)
        eigvals = np.diag(Q.T @ cov @ Q)
        proj = (t @ Q).astype(np.float32)
        Q_f32 = Q.astype(np.float32)
        eigvals_f32 = eigvals.astype(np.float32)
        meta = dict(
            shape=orig_shape,
            m=m,
            n=n,
            d=d,
        )
        data = struct.pack("<III", m, n, d)
        data += _serialize(proj)
        data += _serialize(Q_f32)
        data += _serialize(eigvals_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n, d = struct.unpack_from("<III", data, 0)
        pos = 12
        proj = _deserialize(data[pos : pos + m * d * 4]).reshape(m, d)
        pos += m * d * 4
        Q = _deserialize(data[pos : pos + n * d * 4]).reshape(n, d)
        pos += n * d * 4
        _eigvals = _deserialize(data[pos : pos + d * 4])
        recon = proj @ Q.T
        return recon.reshape(shape).astype(np.float32)

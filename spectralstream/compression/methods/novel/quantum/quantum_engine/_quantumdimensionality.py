from __future__ import annotations

import math
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumDimensionality:
    """Quantum dimensionality reduction via random projections
    (Johnson-Lindenstrauss-like). Use quantum random features.
    """

    name = "quantum_dimensionality"
    category = "quantum_engine"

    def compress(
        self,
        tensor: np.ndarray,
        n_components: int = 8,
        n_projections: int = 64,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        d = min(n_components, n)
        p = min(n_projections, n)
        rng = np.random.RandomState(42)
        proj_matrix = rng.randn(p, n).astype(np.float64) / math.sqrt(p)
        features = proj_matrix @ t
        U, S, Vt = np.linalg.svd(
            features.reshape(-1, max(1, p // d)), full_matrices=False
        )
        k = min(d, len(S))
        components = (U[:, :k] * S[:k]).astype(np.float32)
        proj_f32 = proj_matrix.astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            p=p,
            k=k,
        )
        data = struct.pack("<III", n, p, k)
        data += _serialize(proj_f32)
        data += _serialize(components)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, p, k = struct.unpack_from("<III", data, 0)
        pos = 12
        proj_matrix = _deserialize(data[pos : pos + p * n * 4]).reshape(p, n)
        pos += p * n * 4
        components = _deserialize(data[pos : pos + (p // max(1, p // k)) * k * 4])
        recon = np.linalg.pinv(proj_matrix) @ np.pad(
            components, (0, p - len(components)), mode="constant"
        )
        return recon.reshape(shape).astype(np.float32)

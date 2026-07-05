from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SingularValueDensity:
    name = "singular_value_density"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            u, s, vt = np.linalg.svd(t, full_matrices=False)
            if rank is None:
                cum = np.cumsum(s**2) / np.sum(s**2)
                rank = max(1, int(np.searchsorted(cum, 0.9)) + 1)
            rank = min(rank, len(s))
            s_rest = s[rank:]
            mu = float(np.mean(s_rest)) if len(s_rest) > 0 else 0.0
            sigma = float(np.std(s_rest)) if len(s_rest) > 0 else 0.0
            skew = (
                float(np.mean((s_rest - mu) ** 3) / (sigma**3 + 1e-30))
                if len(s_rest) > 1
                else 0.0
            )
            meta = dict(
                shape=orig_shape,
                rank=rank,
                dist_mu=mu,
                dist_sigma=sigma,
                dist_skew=skew,
            )
            data = struct.pack("<ifff", rank, mu, sigma, skew)
            data += _serialize(u[:, :rank].astype(np.float32))
            data += _serialize(s[:rank].astype(np.float32))
            data += _serialize(vt[:rank, :].astype(np.float32))
            return data, meta
        meta = dict(
            shape=orig_shape, rank=0, dist_mu=0.0, dist_sigma=0.0, dist_skew=0.0
        )
        data = struct.pack("<ifff", 0, 0.0, 0.0, 0.0) + _serialize(
            t.ravel().astype(np.float32)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank, mu, sigma, skew = struct.unpack_from("<ifff", data, 0)
        pos = struct.calcsize("<ifff")
        if rank == 0:
            flat = _deserialize(data[pos:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        m, n = shape
        u = _deserialize(data[pos : pos + m * rank * 4]).reshape(m, rank)
        pos += m * rank * 4
        s_top = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        vt = _deserialize(data[pos : pos + rank * n * 4]).reshape(rank, n)
        return ((u * s_top) @ vt).astype(np.float32)

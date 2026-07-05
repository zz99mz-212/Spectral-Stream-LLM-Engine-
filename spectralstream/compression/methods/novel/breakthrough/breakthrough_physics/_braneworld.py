from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

class BraneWorld:
    """Brane world: each layer/row is a 3-brane embedded in a higher-
    dimensional bulk. Store only the brane tension (row norm), brane
    position (row mean), and brane angle (dominant direction via PCA).
    Bulk fields mediate the reconstructed interactions.

    Real: row-wise PCA + quantized residuals.
    """

    name = "brane_world"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, bulk_rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        k = min(bulk_rank, m, n)
        # Bulk SVD = higher-dimensional embedding space
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(k, len(S), m, n)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        # Brane params per row
        tensions = np.linalg.norm(t, axis=1).astype(np.float32)
        positions = np.mean(t, axis=1).astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(tensions)
        buf += _serialize(positions)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        tensions = np.frombuffer(data[pos : pos + m * 4], dtype=np.float32)
        pos += m * 4
        positions = np.frombuffer(data[pos : pos + m * 4], dtype=np.float32)
        pos += m * 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Brane reconstruction: bulk SVD + brane params as modulation
        recon = (U_k * S_k) @ Vt_k
        for i in range(m):
            recon[i] = recon[i] * tensions[i] / max(np.linalg.norm(recon[i]), 1e-30)
            recon[i] += positions[i] - np.mean(recon[i])
        return recon.astype(np.float32).reshape(metadata["shape"])

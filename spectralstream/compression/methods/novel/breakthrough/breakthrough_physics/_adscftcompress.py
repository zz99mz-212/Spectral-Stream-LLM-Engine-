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

class AdSCFTCompress:
    """AdS/CFT holography: the weight matrix is a boundary CFT correlator.
    Store the bulk gravitational theory (defined by truncated SVD
    of the matrix — the 'boundary data'). The bulk geometry is the
    entanglement structure of the SVD components.

    Reconstruction: boundary data → bulk reconstruction via
    'holographic renormalization' (SVD reconstruction).
    """

    name = "ads_cft_compress"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 24) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        # Boundary CFT = the weight matrix itself
        # Bulk = truncated SVD (the holographic dual)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # AdS radius = scale of singular values
        ads_radius = float(np.mean(S[:k]))
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<f", ads_radius)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "ads_radius": ads_radius,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        R = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Holographic reconstruction: boundary CFT from bulk AdS
        recon = (U_k * (S_k * R / max(R, 1e-10))) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])

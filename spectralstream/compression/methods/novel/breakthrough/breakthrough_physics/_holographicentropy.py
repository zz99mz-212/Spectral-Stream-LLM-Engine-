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

class HolographicEntropy:
    """Holographic entanglement entropy: weight information is stored in
    Ryu-Takayanagi minimal surfaces in the holographic bulk. Different
    weights correspond to RT surfaces with area proportional to
    entanglement entropy S = Area(γ_A) / 4G_N.

    Real: SVD components are 'RT surfaces' — their singular values
    correspond to surface areas (entropy). Store top-k surfaces.
    """

    name = "holographic_entropy"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Entropy spectrum = log of singular values
        entropy_spectrum = np.log(S[:k] + 1e-30).astype(np.float32)
        U_k = U[:, :k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(entropy_spectrum)
        buf += _serialize(U_k)
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
        entropy = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # RT surface area → singular values via S = exp(entropy)
        S_k = np.exp(entropy)
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])

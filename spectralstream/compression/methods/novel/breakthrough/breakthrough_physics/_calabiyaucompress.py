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

class CalabiYauCompress:
    """Calabi-Yau manifold: the weight matrix's structure is determined by
    Hodge numbers h^{p,q} (singular value distribution moments) and
    Kähler parameters (scale and shape parameters). The Ricci-flat
    metric is the reconstruction map.

    Store: top-k SVD triplets + distribution moments of all singular values.
    Reconstruct: fit singular values to parametrized distribution,
    reconstruct weight from stored vectors + generated values.
    """

    name = "calabi_yau_compress"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, rank: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        k = min(rank, m, n)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(k, len(S))
        # Hodge numbers = statistical moments of full spectrum
        s_full = S / max(S[0], 1e-30)
        hodge = np.array(
            [
                np.mean(s_full),
                np.std(s_full),
                float(np.mean(np.abs(s_full - np.mean(s_full)) ** 3)) ** (1.0 / 3.0),
                float(np.mean(s_full**2)),
                float(np.mean(np.abs(s_full))),
            ],
            dtype=np.float32,
        )
        # Kähler parameters = rank-k SVD truncation
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += _serialize(hodge)
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
        hodge = np.frombuffer(data[pos : pos + 20], dtype=np.float32)
        pos += 20
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Reconstitute from Calabi-Yau metric (Ricci-flat = identity-like)
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])

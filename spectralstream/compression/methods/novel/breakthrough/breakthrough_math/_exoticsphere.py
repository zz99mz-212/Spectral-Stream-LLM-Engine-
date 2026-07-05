from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class ExoticSphere:
    """Exotic spheres: weight matrices live on exotic differential
    structures (Milnor spheres). The 7-sphere S^7 has 28 oriented
    diffeomorphism classes. Store the exotic smoothness parameter
    (index k ∈ {0,...,27} for S^7) which determines the mapping
    between singular vectors via the exotic framing.

    Real: the 'exotic structure' is the permutation of singular
    vectors induced by the exotic framing map. Store the permutation
    index + top-k SVD components.
    """

    name = "exotic_sphere"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 12) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Exotic class = permutation of singular vectors
        # determined by the spectral distribution pattern
        s_norm = S / max(S[0], 1e-30)
        # Map spectral shape to exotic class (0-27 for S^7)
        spectral_moment = float(np.mean(s_norm[: min(8, len(s_norm))] ** 2))
        exotic_class = int(spectral_moment * 27) % 28
        # Apply exotic framing permutation
        perm = np.arange(k)
        if k >= 7:
            rng = np.random.RandomState(exotic_class)
            rng.shuffle(perm)
        U_k = U[:, perm[:k]].copy().astype(np.float32)
        S_k = S[perm[:k]].astype(np.float32)
        Vt_k = Vt[perm[:k], :].copy().astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<B", exotic_class)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "exotic_class": exotic_class,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        exotic_class = struct.unpack_from("<B", data, pos)[0]
        pos += 1
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Reverse exotic framing
        perm = np.arange(k)
        if k >= 7:
            rng = np.random.RandomState(exotic_class)
            inv_perm = np.arange(k)
            rng.shuffle(perm)
            inv_perm[perm] = np.arange(k)
            U_k = U_k[:, inv_perm]
            S_k = S_k[inv_perm]
            Vt_k = Vt_k[inv_perm, :]
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])

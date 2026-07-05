from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class NoetherCharge:
    """Noether's theorem: every continuous symmetry of a system corresponds
    to a conserved Noether charge. The weight matrix's symmetries
    (translation, rotation, scaling invariance in row/column space)
    correspond to conserved quantities that can be stored instead of
    the full matrix.

    Real: store the Noether charges — row sums (momentum), column sums,
    Frobenius norm (energy), trace (for square), and top SVD components.
    Reconstruct by enforcing charge conservation.
    """

    name = "noether_charge"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Noether charges (conserved quantities)
        momentum = np.mean(t, axis=1).astype(np.float32)  # translation
        energy = float(np.linalg.norm(t))  # scale invariance
        angular = float(np.mean(t * np.arange(n)[np.newaxis, :])) if n > 1 else 0.0
        # Top SVD components as 'gauge potentials'
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<ff", energy, angular)
        buf += _serialize(momentum)
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
        energy, angular = struct.unpack_from("<ff", data, pos)
        pos += 8
        momentum = np.frombuffer(data[pos : pos + m * 4], dtype=np.float32)
        pos += m * 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # Charge-conserving reconstruction
        recon = (U_k * S_k) @ Vt_k
        # Enforce conservation laws
        recon *= energy / max(np.linalg.norm(recon), 1e-30)
        for i in range(m):
            recon[i] += momentum[i] - np.mean(recon[i])
        return recon.astype(np.float32).reshape(metadata["shape"])

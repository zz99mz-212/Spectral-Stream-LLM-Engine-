from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class HodgeDuality:
    """Hodge star operator: every k-form has a Hodge dual (n-k)-form.
    Decompose the weight matrix into its harmonic (∇²φ=0), exact
    (dα), and co-exact (δβ) components via the Hodge decomposition.
    The harmonic part is fully determined by boundary conditions;
    store only the exact and co-exact parts compactly.

    Real: additive decomposition into symmetric (exact), anti-
    symmetric (co-exact), and trace (harmonic) parts.
    Store the harmonic part (trace) + compressed symmetric part.
    """

    name = "hodge_duality"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 12) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Hodge decomposition
        harmonic = float(np.trace(t)) if m == n else float(np.mean(t))
        exact = (t + t.T) / 2.0 if m == n else t.copy()
        co_exact = (t - t.T) / 2.0 if m == n else np.zeros_like(t)
        # Compress exact (symmetric) part via SVD
        U_e, S_e, Vt_e = np.linalg.svd(exact, full_matrices=False)
        k_e = min(k, len(S_e))
        U_k = U_e[:, :k_e].astype(np.float32)
        S_k = S_e[:k_e].astype(np.float32)
        Vt_k = Vt_e[:k_e, :].astype(np.float32)
        # Compress co-exact (anti-symmetric) as sparse
        c_mask = np.abs(co_exact) > np.percentile(np.abs(co_exact), 85)
        c_idx = np.where(c_mask.ravel())[0].astype(np.int32)
        c_vals = co_exact.ravel()[c_mask].astype(np.float16)
        buf = struct.pack("<III", m, n, k_e)
        buf += struct.pack("<dI", harmonic, len(c_idx))
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        buf += c_idx.tobytes()
        buf += c_vals.tobytes()
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k_e,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        harmonic, n_co = struct.unpack_from("<dI", data, pos)
        pos += 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        pos += k * n * 4
        exact = (U_k * S_k) @ Vt_k
        if n_co > 0:
            c_idx = np.frombuffer(data[pos : pos + n_co * 4], dtype=np.int32)
            pos += n_co * 4
            c_vals = np.frombuffer(data[pos : pos + n_co * 2], dtype=np.float16).astype(
                np.float64
            )
            co_exact = np.zeros(m * n, dtype=np.float64)
            for idx, val in zip(c_idx, c_vals):
                if 0 <= idx < m * n:
                    co_exact[idx] = val
            co_exact = co_exact.reshape(m, n)
            recon = exact + co_exact
        else:
            recon = exact
        if m == n:
            recon += np.eye(m) * harmonic / max(m, 1)
        return recon.astype(np.float32).reshape(metadata["shape"])

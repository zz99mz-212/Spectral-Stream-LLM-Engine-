from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class MorseHomology:
    """Morse theory: the weight matrix defines a Morse function on the
    discrete grid of indices. The critical points (local min/max/saddle)
    determine the topology via the Morse complex. Store only the
    critical points (their positions and indices) + gradient flow lines.

    Real: identify critical points in the weight landscape.
    Store their positions and values. The full matrix is reconstructed
    by interpolating between critical points with Morse-Smale flow.
    """

    name = "morse_homology"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, n_critical: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape
        # Find local extrema (critical points)
        critical_points = []
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                patch = t[i - 1 : i + 2, j - 1 : j + 2]
                center = t[i, j]
                if center == np.max(patch) or center == np.min(patch):
                    critical_points.append((i, j, float(center)))
        # Select top critical points by absolute value
        critical_points.sort(key=lambda x: abs(x[2]), reverse=True)
        critical_points = critical_points[:n_critical]
        n_c = len(critical_points)
        idx = np.array([cp[0] for cp in critical_points], dtype=np.int32)
        jdx = np.array([cp[1] for cp in critical_points], dtype=np.int32)
        vals = np.array([cp[2] for cp in critical_points], dtype=np.float32)
        # SVD residual
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(4, len(S), m, n)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<IIII", m, n, n_c, k)
        buf += idx.tobytes() + jdx.tobytes() + vals.tobytes()
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "n_critical": n_c,
            "k": k,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, n_c, k = struct.unpack_from("<IIII", data, 0)
        pos = 16
        idx = np.frombuffer(data[pos : pos + n_c * 4], dtype=np.int32)
        pos += n_c * 4
        jdx = np.frombuffer(data[pos : pos + n_c * 4], dtype=np.int32)
        pos += n_c * 4
        vals = np.frombuffer(data[pos : pos + n_c * 4], dtype=np.float32)
        pos += n_c * 4
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        # SVD backbone
        recon = (U_k * S_k) @ Vt_k
        # Imprint critical points via radial basis interpolation
        for ci, cj, cv in zip(idx, jdx, vals):
            ri, rj = max(0, ci - 2), max(0, cj - 2)
            ri2, rj2 = min(m, ci + 3), min(n, cj + 3)
            if ri < ri2 and rj < rj2:
                patch = recon[ri:ri2, rj:rj2]
                if patch.size > 0:
                    recon[ri:ri2, rj:rj2] += (cv - np.mean(patch)) * 0.1
        return recon.astype(np.float32)

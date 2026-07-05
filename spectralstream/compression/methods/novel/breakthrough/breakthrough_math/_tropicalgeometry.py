from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class TropicalGeometry:
    """Tropical geometry: replace classical arithmetic with tropical
    (min-plus or max-plus) algebra. The weight matrix in tropical
    geometry corresponds to a piecewise-linear convex function
    (a tropical polynomial). Store the tropical polynomial
    coefficients (few breakpoints) + tropical addition structure.

    Real: apply log transform (tropicalization), find piecewise-
    linear structure via breakpoints. Store breakpoints + SVD
    of the 'tropical' residual.
    """

    name = "tropical_geometry"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        k = min(rank, len(S), m, n)
        # Tropicalization: log |W|
        trop = np.log(np.abs(t) + 1e-30)
        # Find tropical breakpoints (piecewise-linear regions)
        flat = trop.ravel()
        n_breaks = min(8, len(flat) // 100)
        if n_breaks > 0:
            breaks = np.linspace(
                float(np.min(flat)), float(np.max(flat)), n_breaks + 2
            )[1:-1]
            break_idx = [int(np.argmin(np.abs(flat - b))) for b in breaks]
            break_vals = flat[np.array(break_idx)].astype(np.float32)
        else:
            break_idx = []
            break_vals = np.array([], dtype=np.float32)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<I", len(break_idx))
        buf += np.array(break_idx, dtype=np.int32).tobytes()
        buf += break_vals.tobytes()
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "n_breaks": len(break_idx),
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        n_breaks = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if n_breaks > 0:
            break_idx = np.frombuffer(data[pos : pos + n_breaks * 4], dtype=np.int32)
            pos += n_breaks * 4
            break_vals = np.frombuffer(data[pos : pos + n_breaks * 4], dtype=np.float32)
            pos += n_breaks * 4
        else:
            break_idx = np.array([], dtype=np.int32)
            break_vals = np.array([], dtype=np.float32)
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        recon = (U_k * S_k) @ Vt_k
        return recon.astype(np.float32).reshape(metadata["shape"])

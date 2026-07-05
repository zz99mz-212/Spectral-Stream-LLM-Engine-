from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class BCMRule:
    """D3. BCM-RULE: Δw_ij = y_i(y_j - θ_M)w_ij with sliding threshold θ_M."""

    name = "bcm_rule"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, n_keep: int = None) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        y_i = np.mean(t, axis=1)
        y_j = np.mean(t, axis=0)
        theta_m = np.mean(t)

        importance = (
            np.abs(t) * np.abs(y_i[:, None] - theta_m) * np.abs(y_j[None, :] - theta_m)
        )
        importance_flat = importance.ravel()

        if n_keep is None:
            n_keep = max(1, int(0.15 * len(importance_flat)))

        idx = np.argpartition(importance_flat, -n_keep)[-n_keep:]
        vals = t.ravel()[idx]

        meta = dict(shape=t.shape, n_keep=n_keep, theta_m=float(theta_m))
        data = _serialize(idx.astype(np.int32)) + vals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_keep = metadata["n_keep"]
        N = int(np.prod(shape))

        idx = _deserialize(data[: n_keep * 4]).astype(int)
        vals = np.frombuffer(data[n_keep * 4 :], dtype=np.float16).astype(np.float64)

        recon = np.full(N, float(metadata["theta_m"]), dtype=np.float64)
        for i in range(min(n_keep, len(idx), len(vals))):
            if idx[i] < N:
                recon[idx[i]] = vals[i]

        return recon.reshape(shape).astype(np.float32)

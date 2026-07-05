from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class HomeostaticScale:
    """D2. HOMEOSTATIC-SCALE: τ dw/dt = w₀ - w, synaptic scaling."""

    name = "homeostatic_scale"
    category = "novel_biological"

    def compress(
        self, tensor: np.ndarray, target_mean: float = 0.0, tau: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)

        scaled = t + (target_mean - np.mean(t)) * (1.0 - np.exp(-1.0 / tau))

        U, S, Vt = np.linalg.svd(scaled, full_matrices=False)
        cum = np.cumsum(S) / np.sum(S)
        r = int(np.searchsorted(cum, 0.90)) + 1
        r = min(r, len(S))

        meta = dict(shape=t.shape, r=int(r), target_mean=target_mean, tau=tau)
        data = (
            _serialize(U[:, :r].astype(np.float32))
            + _serialize(S[:r].astype(np.float32))
            + _serialize(Vt[:r, :].astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r = metadata["r"]
        target_mean = metadata["target_mean"]
        m, n = shape

        U_r = _deserialize(data[: m * r * 4]).reshape(m, r)
        pos = m * r * 4
        S_r = _deserialize(data[pos : pos + r * 4])
        pos += r * 4
        Vt_r = _deserialize(data[pos : pos + r * n * 4]).reshape(r, n)

        recon = (U_r * S_r) @ Vt_r
        recon = recon + (target_mean - np.mean(recon))
        return recon.astype(np.float32)

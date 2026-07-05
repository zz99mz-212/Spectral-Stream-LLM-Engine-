"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class LandauZener:
    """Landau-Zener transition — adiabatic interpolation between states."""

    name = "landau_zener"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        gap = min(params.get("gap", 0.5), 1.0)
        n_samples = min(params.get("n_samples", 16), n // 4)
        idx = np.linspace(0, n - n_samples, n_samples).astype(int)
        states = np.array([flat[i : i + n_samples] for i in idx])
        mean_state = np.mean(states, axis=0)
        meta = dict(n=n, n_samples=n_samples, shape=t.shape)
        data = _serialize(mean_state.astype(np.float32))
        data += _serialize(flat[idx].astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        n_samples = metadata["n_samples"]
        shape = metadata["shape"]
        mean_state = _deserialize(data[: n_samples * 4])
        pos = n_samples * 4
        idx_vals = _deserialize(data[pos : pos + n_samples * 4])
        recon = np.zeros(n, dtype=np.float64)
        recon[:n_samples] = mean_state
        for i in range(n_samples):
            if i < n:
                recon[i] = idx_vals[i]
        for i in range(n_samples, n):
            recon[i] = 0.5 * (mean_state[i % n_samples] + idx_vals[i % n_samples])
        return recon.reshape(shape).astype(np.float32)




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

class SIRENINR:
    """SIREN implicit neural representation — sine-activated MLP encodes weights."""

    name = "siren_inr"
    category = "functional"

    def __init__(self, hidden_dim: int = 32, n_layers: int = 3):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        rng = np.random.RandomState(42)
        rng_j = np.random.RandomState(7)
        coords = np.linspace(-1, 1, n).astype(np.float32)
        hidden = int(params.get("hidden_dim", self.hidden_dim))
        n_l = int(params.get("n_layers", self.n_layers))
        w1 = (rng.randn(1, hidden) * 0.1).astype(np.float32)
        b1 = (rng.randn(hidden) * 0.1).astype(np.float32)
        w_mid = []
        b_mid = []
        for _ in range(n_l - 1):
            w_mid.append((rng_j.randn(hidden, hidden) * 0.1).astype(np.float32))
            b_mid.append((rng_j.randn(hidden) * 0.1).astype(np.float32))
        wo = (rng_j.randn(hidden) * 0.1).astype(np.float32)
        bo = (rng.randn(1) * 0.1).astype(np.float32)
        lr = 0.01
        x = coords.reshape(-1, 1)
        for epoch in range(200):
            h = np.sin(x @ w1 + b1)
            for wi, bi in zip(w_mid, b_mid):
                h = np.sin(h @ wi + bi)
            pred = (h @ wo + bo).ravel()
            loss = pred - flat
            grad_wo = h.T @ loss / n
            grad_bo = float(np.mean(loss))
            wo -= lr * grad_wo.astype(np.float32)
            bo -= lr * grad_bo
            if epoch % 50 == 49:
                lr *= 0.5
        h = np.sin(x @ w1 + b1)
        for wi, bi in zip(w_mid, b_mid):
            h = np.sin(h @ wi + bi)
        pred = (h @ wo + bo).ravel()
        meta = dict(hidden_dim=hidden, n_layers=n_l, n=n, shape=t.shape)
        data = b""
        for arr in [w1, b1, wo, np.array([bo])] + w_mid + b_mid:
            data += _serialize(arr)
        for i, arr in enumerate(w_mid):
            data += _serialize(arr)
        for i, arr in enumerate(b_mid):
            data += _serialize(arr)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        hidden = metadata["hidden_dim"]
        n_l = metadata["n_layers"]
        n = metadata["n"]
        shape = metadata["shape"]
        pos = 0
        w1_s = hidden
        w1 = _deserialize(data[pos : pos + w1_s * 4]).reshape(1, hidden)
        pos += w1_s * 4
        b1 = _deserialize(data[pos : pos + hidden * 4])
        pos += hidden * 4
        wo = _deserialize(data[pos : pos + hidden * 4])
        pos += hidden * 4
        bo = float(_deserialize(data[pos : pos + 4])[0])
        pos += 4
        w_mid = []
        b_mid = []
        for _ in range(n_l - 1):
            w = _deserialize(data[pos : pos + hidden * hidden * 4]).reshape(
                hidden, hidden
            )
            pos += hidden * hidden * 4
            w_mid.append(w)
            b = _deserialize(data[pos : pos + hidden * 4])
            pos += hidden * 4
            b_mid.append(b)
        coords = np.linspace(-1, 1, n).astype(np.float32).reshape(-1, 1)
        h = np.sin(coords @ w1 + b1)
        for wi, bi in zip(w_mid, b_mid):
            h = np.sin(h @ wi + bi)
        result = (h @ wo + bo).ravel()[:n].reshape(shape)
        return result.astype(np.float32)




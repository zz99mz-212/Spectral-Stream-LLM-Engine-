from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasmaEcho:
    """Third-order nonlinear plasma echo: f(t) = Σ A_{k1}B_{k2}exp(i(k1+k2)x - i(ω1+ω2)t)."""

    name = "plasma_echo"
    category = "novel_physics"

    def compress(self, tensor: np.ndarray, n_coupling: int = 12) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        F = np.fft.fft2(t)
        flat = F.ravel()
        k = max(1, min(n_coupling, len(flat) // 4))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        top_modes = flat[idx]

        A = top_modes[: k // 2] if k >= 2 else top_modes
        B = top_modes[k // 2 : 2 * (k // 2)] if k >= 2 else top_modes

        n_c = min(n_coupling, len(A) * len(B))
        rng = np.random.RandomState(42)
        pairs = rng.randint(0, min(len(A), len(B)), (n_c, 2))
        coupling = A[pairs[:, 0]] * B[pairs[:, 1]]

        meta = dict(shape=tensor.shape, k=k, n_c=n_c)
        data = _serialize(idx[:k].astype(np.int32))
        data += top_modes[:k].astype(np.complex64).tobytes()
        data += (
            _serialize(pairs.astype(np.int16)) + coupling.astype(np.complex64).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        n_c = metadata["n_c"]
        m, n = shape

        pos = 0
        idx = _deserialize(data[: k * 4]).astype(int)
        pos += k * 4
        top_modes = np.frombuffer(data[pos : pos + k * 8], dtype=np.complex64).astype(
            np.complex128
        )
        pos += k * 8
        pairs = _deserialize(data[pos : pos + n_c * 4]).reshape(-1, 2).astype(int)
        pos += n_c * 4
        coupling = np.frombuffer(data[pos : pos + n_c * 8], dtype=np.complex64).astype(
            np.complex128
        )

        F = np.zeros(m * n, dtype=np.complex128)
        for i, v in zip(idx, top_modes):
            if i < m * n:
                F[i] = v

        for (p1, p2), c in zip(pairs, coupling):
            if p1 < k and p2 < k and idx[p1] < m * n and idx[p2] < m * n:
                Fi = (idx[p1] + idx[p2]) % (m * n)
                F[Fi] += c

        return np.fft.ifft2(F.reshape(m, n)).real.astype(np.float32)

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumSVM:
    """Quantum SVM: least-squares formulation with quantum kernel.
    Solve via quantum-inspired linear system.
    """

    name = "quantum_svm"
    category = "quantum_compression"

    def _rbf_kernel(self, x: np.ndarray, y: np.ndarray, gamma: float = 0.1) -> float:
        return float(math.exp(-gamma * np.linalg.norm(x - y) ** 2))

    def compress(
        self,
        tensor: np.ndarray,
        gamma: float = 0.1,
        ridge: float = 0.01,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(16, n // 2)
        support_idx = np.linspace(0, n - 1, k, dtype=int)
        support_vecs = t[support_idx]
        K = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                K[i, j] = self._rbf_kernel(
                    support_vecs, np.roll(support_vecs, j), gamma
                )
        y = np.sign(np.random.randn(k))
        alpha = np.linalg.solve(K + ridge * np.eye(k), y)
        b = float(np.mean(y - K @ alpha))
        proj = np.zeros(n, dtype=np.float64)
        for i in range(n):
            k_vals = np.array(
                [
                    self._rbf_kernel(t[i : i + 1], support_vecs[j : j + 1], gamma)
                    for j in range(k)
                ]
            )
            proj[i] = float(k_vals @ alpha + b)
        residual = t - proj
        alpha_f32 = alpha.astype(np.float32)
        sv_f32 = support_vecs.astype(np.float32)
        residual_f32 = residual.astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            k=k,
            gamma=gamma,
            b=b,
        )
        data = struct.pack("<Iff", k, gamma, b)
        data += _serialize(alpha_f32)
        data += _serialize(sv_f32)
        data += _serialize(residual_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        k, gamma, b = struct.unpack_from("<Iff", data, 0)
        pos = 12
        alpha = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        sv = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        residual = _deserialize(data[pos:])
        proj = np.zeros(n, dtype=np.float64)
        for i in range(n):
            k_vals = np.array(
                [
                    self._rbf_kernel(np.array([0.0]), sv[j : j + 1], gamma)
                    for j in range(k)
                ]
            )
            proj[i] = float(k_vals @ alpha + b)
        return (proj + residual[:n]).reshape(shape).astype(np.float32)

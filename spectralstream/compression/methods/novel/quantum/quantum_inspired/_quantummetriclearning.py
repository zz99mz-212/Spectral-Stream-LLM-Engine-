from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumMetricLearning:
    """Quantum metric: Fubini-Study distance between weight states.
    d_FS(ψ, φ) = arccos(|⟨ψ|φ⟩|). Use distance matrix for compressed rep.
    """

    name = "quantum_metric_learning"
    category = "quantum_compression"

    def compress(
        self, tensor: np.ndarray, n_landmarks: int = 8, n_components: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(n_landmarks, n)
        d = min(n_components, k)
        idx = np.linspace(0, n - 1, k, dtype=int)
        landmarks = t[idx]
        D = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            for j in range(k):
                l_val = float(landmarks[j])
                overlap = abs(np.dot(t, t)) / (
                    max(float(np.linalg.norm(t)), 1e-30)
                    * max(float(np.linalg.norm(t)), 1e-30)
                )
                D[i, j] = float(np.arccos(np.clip(overlap, -1.0, 1.0)))
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        proj = (U[:, :d] * S[:d]).astype(np.float32)
        Vt_d = Vt[:d, :].astype(np.float32)
        landmarks_f32 = landmarks.astype(np.float32)
        S_f32 = S[:d].astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            k=k,
            d=d,
            landmark_idx=idx.astype(np.int32).tobytes(),
        )
        data = struct.pack("<III", n, k, d)
        data += idx.astype(np.int32).tobytes()
        data += _serialize(proj)
        data += _serialize(Vt_d)
        data += _serialize(S_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, k, d = struct.unpack_from("<III", data, 0)
        pos = 12
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32)
        pos += k * 4
        proj = _deserialize(data[pos : pos + n * d * 4]).reshape(n, d)
        pos += n * d * 4
        Vt = _deserialize(data[pos : pos + d * k * 4]).reshape(d, k)
        pos += d * k * 4
        S = _deserialize(data[pos : pos + d * 4])
        D_recon = proj @ Vt
        recon = np.zeros(n, dtype=np.float64)
        for i in range(n):
            recon[i] = float(np.cos(D_recon[i].mean()))
        return recon.reshape(shape).astype(np.float32)

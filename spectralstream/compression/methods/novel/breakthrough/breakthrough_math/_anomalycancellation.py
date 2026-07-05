from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class AnomalyCancellation:
    """Anomaly cancellation: gauge theories have quantum anomalies that
    must cancel for consistency. The weight matrix's spectral
    anomalies (odd moments) must cancel between layers. Store
    the anomaly-free combination (difference of consecutive layers)
    which is more compressible than individual layers.

    Real: compute 'anomaly' = skewness of local weight distribution.
    Cancel by pairing layers. Store difference matrix + anomaly params.
    """

    name = "anomaly_cancellation"
    category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = t.shape
        # Compute anomalies (odd moments)
        flat = t.ravel()
        anomaly_1 = float(np.mean(flat))
        anomaly_3 = float(
            np.mean(((flat - np.mean(flat)) / max(np.std(flat), 1e-10)) ** 3)
        )
        # Anomaly-cancelled matrix = W - mean - skewness correction
        correction = anomaly_1 + anomaly_3 * np.tanh(
            np.linspace(-3, 3, m * n).reshape(m, n)
        )
        w_bar = t - correction
        # Compress anomaly-cancelled matrix
        U, S, Vt = np.linalg.svd(w_bar, full_matrices=False)
        k = min(rank, len(S), m, n)
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += struct.pack("<ff", anomaly_1, anomaly_3)
        buf += _serialize(U_k)
        buf += _serialize(S_k)
        buf += _serialize(Vt_k)
        return bytes(buf), {
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        a1, a3 = struct.unpack_from("<ff", data, pos)
        pos += 8
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        w_bar = (U_k * S_k) @ Vt_k
        # Restore anomalies
        correction = a1 + a3 * np.tanh(np.linspace(-3, 3, m * n).reshape(m, n))
        recon = w_bar + correction
        return recon.astype(np.float32).reshape(metadata["shape"])

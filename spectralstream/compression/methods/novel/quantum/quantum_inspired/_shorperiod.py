from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class ShorPeriod:
    """Shor-inspired period finding in weight distributions.
    QFT + continued fractions to find repeating patterns in weight values.
    """

    name = "shor_period"
    category = "quantum_compression"

    def compress(self, tensor: np.ndarray, max_period: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        p = min(max_period, n // 2)
        autocorr = np.zeros(p)
        for tau in range(1, p + 1):
            autocorr[tau - 1] = float(np.corrcoef(t[:-tau], t[tau:])[0, 1])
        autocorr = np.nan_to_num(autocorr)
        spectrum = np.abs(np.fft.rfft(autocorr))
        peaks = np.argsort(-spectrum)[:5]
        periods = []
        for peak in peaks:
            if peak > 0:
                cf = peak / len(autocorr)
                period = int(round(1.0 / cf)) if cf > 0 else 1
                if 1 < period < max_period:
                    periods.append(period)
        if not periods:
            periods = [1]
        best_p = periods[0]
        n_periods = math.ceil(n / best_p)
        template = np.zeros(best_p, dtype=np.float64)
        count = np.zeros(best_p, dtype=np.float64)
        for i in range(n):
            template[i % best_p] += t[i]
            count[i % best_p] += 1.0
        template = np.where(count > 0, template / count, 0.0)
        residual = t - np.tile(template, n_periods)[:n]
        template_f32 = template.astype(np.float32)
        residual_f32 = residual.astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            period=best_p,
            n_periods=n_periods,
        )
        data = struct.pack("<II", best_p, n_periods)
        data += _serialize(template_f32)
        data += _serialize(residual_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        best_p, n_periods = struct.unpack_from("<II", data, 0)
        pos = 8
        template = _deserialize(data[pos : pos + best_p * 4])
        pos += best_p * 4
        residual = _deserialize(data[pos:])
        periodic = np.tile(template, n_periods)[:n]
        recon = periodic + residual
        return recon.reshape(shape).astype(np.float32)

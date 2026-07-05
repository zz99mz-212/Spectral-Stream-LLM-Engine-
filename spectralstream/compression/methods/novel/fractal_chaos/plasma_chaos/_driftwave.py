from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class DriftWave:
    """Drift wave turbulence: W = Σ φ_k exp(ik·x - iω_k t), keep unstable modes only."""

    name = "drift_wave"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, keep_fraction: float = 0.15
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        F = np.fft.fft2(t)
        shifted = np.fft.fftshift(F)
        amplitude = np.abs(shifted)
        phase = np.angle(shifted)

        flat_amp = amplitude.ravel()
        k = max(1, int(keep_fraction * flat_amp.size))
        idx = np.argpartition(-flat_amp, k)[:k]

        growth_rates = np.gradient(np.log(flat_amp[idx] + 1e-30))
        unstable_mask = growth_rates > np.median(growth_rates)
        kept_idx = idx[unstable_mask]
        if len(kept_idx) == 0:
            kept_idx = idx[: max(1, k // 4)]

        rows, cols = np.unravel_index(kept_idx, amplitude.shape)
        kept_amp = amplitude[rows, cols]
        kept_phase = phase[rows, cols]

        meta = dict(shape=tensor.shape, n_kept=len(kept_idx))
        data = _serialize(np.stack([rows, cols], axis=1).astype(np.int32))
        data += _serialize(kept_amp.astype(np.float16))
        data += kept_phase.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        m, n = shape

        pos = 0
        rc_data = _deserialize(data[: n_kept * 8]).reshape(-1, 2).astype(int)
        pos += n_kept * 8
        amp = np.frombuffer(data[pos : pos + n_kept * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_kept * 2
        phase = np.frombuffer(data[pos : pos + n_kept * 2], dtype=np.float16).astype(
            np.float64
        )

        F = np.zeros((m, n), dtype=np.complex128)
        for (r, c), a, p in zip(rc_data, amp, phase):
            if r < m and c < n:
                F[r, c] = a * np.exp(1j * p)
                F[(m - r) % m, (n - c) % n] = (
                    np.conj(F[r, c]) if (r != 0 and c != 0) else F[r, c]
                )

        return np.fft.ifft2(F).real.astype(np.float32)

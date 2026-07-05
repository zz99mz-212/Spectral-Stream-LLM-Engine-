from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasmaHeating:
    """RF heating: ω - k_∥v_∥ = nω_ci resonance layer retention."""

    name = "plasma_heating"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, resonance_keep: float = 0.2
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        F = np.fft.fft2(t)
        shifted = np.fft.fftshift(F)

        kx = np.fft.fftshift(np.fft.fftfreq(m))
        ky = np.fft.fftshift(np.fft.fftfreq(n))
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        omega = np.sqrt(KX**2 + KY**2 + 1e-30)
        k_parallel = np.abs(KX)
        v_parallel = np.linspace(-1, 1, m)[:, None]
        omega_ci = 1.0

        resonance = np.abs(omega - k_parallel * v_parallel - omega_ci)
        resonance_weight = 1.0 / (resonance + 1e-10)
        resonant_mask = resonance_weight > np.percentile(
            resonance_weight, int((1 - resonance_keep) * 100)
        )

        kept_idx = np.argwhere(resonant_mask)
        kept_vals = shifted[resonant_mask]

        meta = dict(shape=tensor.shape, n_kept=int(np.sum(resonant_mask)))
        data = _serialize(kept_idx.astype(np.int32))
        data += kept_vals.astype(np.complex64).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        m, n = shape

        idx = _deserialize(data[: n_kept * 8]).reshape(-1, 2).astype(int)
        vals = np.frombuffer(data[n_kept * 8 :], dtype=np.complex64).astype(
            np.complex128
        )

        F = np.zeros((m, n), dtype=np.complex128)
        for (r, c), v in zip(idx, vals):
            if r < m and c < n:
                F[r, c] = v
                F[(m - r) % m, (n - c) % n] = np.conj(v) if (r != 0 and c != 0) else v

        return np.fft.ifft2(F).real.astype(np.float32)

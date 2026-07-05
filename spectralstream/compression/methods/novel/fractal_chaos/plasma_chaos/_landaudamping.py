from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class LandauDamping:
    """Phase mixing decomposition, Landau damping rate γ determines retention."""

    name = "landau_damping"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, damping_threshold: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        F = np.fft.fft2(t)
        shifted = np.fft.fftshift(F)
        ky = np.fft.fftshift(np.fft.fftfreq(n))
        kx = np.fft.fftshift(np.fft.fftfreq(m))
        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        k_sq = KX**2 + KY**2 + 1e-30
        v_phi = np.sqrt(k_sq)
        f_prime = -np.abs(shifted) / (np.abs(v_phi) + 1e-30)
        gamma = np.pi * f_prime / k_sq

        keep = gamma < damping_threshold
        kept_vals = shifted[keep]
        kept_idx = np.argwhere(keep)

        meta = dict(shape=tensor.shape, n_kept=int(np.sum(keep)))
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

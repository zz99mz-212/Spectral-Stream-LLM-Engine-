from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class PlasmaInstability:
    """ITG criterion η_i > η_crit ≈ 2/3 mode selection."""

    name = "plasma_instability"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, eta_crit: float = 2.0 / 3.0
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
        k_perp = np.sqrt(KX**2 + KY**2)

        T_profile = np.abs(shifted)
        dn_profile = np.abs(np.gradient(np.log(np.abs(shifted) + 1e-30)))
        dT_profile = np.abs(np.gradient(np.log(T_profile + 1e-30)))
        eta = np.sqrt(dT_profile.sum(axis=0, keepdims=True) ** 2 + 1e-30) / (
            np.sqrt(dn_profile.sum(axis=1, keepdims=True) ** 2 + 1e-30) + 1e-30
        )

        unstable = eta > eta_crit
        if np.sum(unstable) < 4:
            k_vals = np.abs(F.ravel())
            unstable = np.zeros(m * n, dtype=bool)
            unstable[np.argpartition(-k_vals, 4)[:4]] = True
            unstable = unstable.reshape(m, n)

        kept_idx = np.argwhere(unstable)
        kept_vals = shifted[unstable]

        meta = dict(shape=tensor.shape, n_kept=int(np.sum(unstable)))
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

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class TokamakFlux:
    """Magnetic flux coordinate transform (ψ,θ,φ), keep low (m,n) modes."""

    name = "tokamak_flux"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, max_m: int = 8, max_n: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        F = np.fft.fft2(t)
        m_modes = min(max_m, m // 2)
        n_modes = min(max_n, n // 2)

        mask = np.zeros((m, n), dtype=bool)
        mask[:m_modes, :n_modes] = True
        mask[-m_modes:, :n_modes] = True
        mask[:m_modes, -n_modes:] = True
        mask[-m_modes:, -n_modes:] = True

        kept = F[mask]
        kept_flat = F.ravel()
        idx = np.argwhere(mask.ravel())[:, 0]

        meta = dict(shape=tensor.shape, n_kept=len(kept), max_m=m_modes, max_n=n_modes)
        data = _serialize(idx.astype(np.int32)) + kept.astype(np.complex64).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        m, n = shape

        idx = _deserialize(data[: n_kept * 4]).astype(int)
        vals = np.frombuffer(data[n_kept * 4 :], dtype=np.complex64).astype(
            np.complex128
        )

        F = np.zeros((m, n), dtype=np.complex128)
        for i, v in zip(idx, vals):
            if i < m * n:
                r, c = divmod(i, n)
                F[r, c] = v

        return np.fft.ifft2(F).real.astype(np.float32)

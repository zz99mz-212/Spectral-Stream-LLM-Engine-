from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class ZonalFlow:
    """Decompose into zonal (k_y=0) + non-zonal components."""

    name = "zonal_flow"
    category = "novel_physics"

    def compress(
        self, tensor: np.ndarray, zonal_keep: float = 0.1
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        zonal = np.mean(t, axis=1, keepdims=True)
        nonzonal = t - zonal

        Uz, Sz, Vtz = np.linalg.svd(zonal, full_matrices=False)
        rz = max(1, int(zonal_keep * min(m, 1)))
        rz = min(rz, len(Sz))

        F_nz = np.fft.fft2(nonzonal)
        shifted_nz = np.fft.fftshift(F_nz)
        flat_nz = shifted_nz.ravel()
        k_nz = max(1, int(0.08 * flat_nz.size))
        idx_nz = np.argpartition(np.abs(flat_nz), -k_nz)[-k_nz:]
        vals_nz = flat_nz[idx_nz]

        meta = dict(shape=tensor.shape, rz=rz, n_nz=len(idx_nz))
        data = _serialize(Uz[:, :rz].astype(np.float32))
        data += _serialize(Sz[:rz].astype(np.float32))
        data += _serialize(Vtz[:rz, :].astype(np.float32))
        data += _serialize(idx_nz.astype(np.int32))
        data += vals_nz.astype(np.complex64).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rz = metadata["rz"]
        n_nz = metadata["n_nz"]

        pos = 0
        Uz = _deserialize(data[: shape[0] * rz * 4]).reshape(shape[0], rz)
        pos += shape[0] * rz * 4
        Sz = _deserialize(data[pos : pos + rz * 4])
        pos += rz * 4
        Vtz = _deserialize(data[pos : pos + rz * 1 * 4]).reshape(rz, 1)
        pos += rz * 1 * 4
        zonal = (Uz * Sz) @ Vtz

        idx_nz = _deserialize(data[pos : pos + n_nz * 4]).astype(int)
        pos += n_nz * 4
        vals_nz = np.frombuffer(data[pos : pos + n_nz * 8], dtype=np.complex64).astype(
            np.complex128
        )

        F_nz = np.zeros(shape, dtype=np.complex128)
        flat_nz = F_nz.ravel()
        for i, v in zip(idx_nz, vals_nz):
            if i < flat_nz.size:
                flat_nz[i] = v

        nonzonal = np.fft.ifft2(F_nz).real
        return (zonal + nonzonal).astype(np.float32)

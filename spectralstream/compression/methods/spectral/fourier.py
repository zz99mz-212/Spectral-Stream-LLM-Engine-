"""HPC Fourier — fully vectorized rfft2/irfft2, zero Python loops."""

from __future__ import annotations

import struct
from typing import Any, Tuple

import numpy as np

from spectralstream.core.math_primitives import auto_keep_fraction


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class Fourier:
    """2D Fourier transform with dominant frequency retention — fully vectorized."""

    name = "fourier"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        F = np.fft.rfft2(orig)
        flat = F.ravel()
        if keep_fraction is None:
            kf = auto_keep_fraction(flat, target_energy)
        else:
            kf = keep_fraction
        k = max(1, int(kf * flat.size))
        mag = np.abs(flat)
        idx = np.argpartition(mag, -k)[-k:]
        meta = dict(
            shape=orig.shape, keep_fraction=kf, target_energy=target_energy, n_kept=k
        )
        data = (
            struct.pack("<ii", *orig.shape)
            + idx.astype(np.int32).tobytes()
            + flat[idx].astype(np.complex64).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n = metadata["shape"]
        k = metadata["n_kept"]
        pos = struct.calcsize("<ii")
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy().astype(int)
        pos += k * 4
        vals = np.frombuffer(data[pos : pos + k * 8], dtype=np.complex64).astype(
            np.complex128
        )
        Fr = np.zeros(m * (n // 2 + 1), dtype=np.complex128)
        Fr[idx] = vals
        return np.fft.irfft2(Fr.reshape(m, n // 2 + 1), s=(m, n)).real.astype(
            np.float32
        )


class FrequencyDomain:
    """General frequency-domain compression with adaptive threshold."""

    name = "frequency_domain"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        from .dct import DCT2D

        return DCT2D().compress(
            tensor, keep_fraction=keep_fraction, target_energy=target_energy
        )

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from .dct import DCT2D

        return DCT2D().decompress(data, metadata)

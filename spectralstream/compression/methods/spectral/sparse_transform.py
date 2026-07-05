"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    auto_keep_fraction,
    dct,
    idct,
    dct_2d,
    idct_2d,
    fwht,
    ifwht,
    next_power_of_two,
    WaveletTransform,
    NTT,
)


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class ButterflySparse:
    """Butterfly sparse transform with coefficient thresholding."""

    name = "butterfly_sparse"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        from .dct import DCTBlock

        return DCTBlock().compress(
            tensor,
            block_size=32,
            keep_fraction=keep_fraction,
            target_energy=target_energy,
        )

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from .dct import DCTBlock

        return DCTBlock().decompress(data, metadata)


class SparseRandomProjection:
    """Sparse random projection (Johnson-Lindenstrauss)."""

    name = "sparse_random_projection"
    category = "spectral"

    def compress(
        self, tensor: np.ndarray, n_components: int = 64
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        m, n = t.shape
        rng = np.random.RandomState(42)
        scale = np.sqrt(3.0 / n_components) if n_components > 0 else 1.0
        R = rng.choice(
            [-scale, 0, scale], size=(n, n_components), p=[1 / 6, 2 / 3, 1 / 6]
        ).astype(np.float64)
        proj = t @ R
        meta = dict(shape=tensor.shape, n_components=n_components)
        data = _serialize(proj.astype(np.float32)) + _serialize(R.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_components = metadata["n_components"]
        m = shape[0] if len(shape) >= 2 else 1
        n = shape[-1] if len(shape) >= 1 else 1
        proj = _deserialize(data[: m * n_components * 4]).reshape(m, n_components)
        R = _deserialize(data[m * n_components * 4 :]).reshape(n, n_components)
        try:
            return (proj @ np.linalg.pinv(R.T)).reshape(shape).astype(np.float32)
        except (np.linalg.LinAlgError, ValueError):
            return np.zeros(shape, dtype=np.float32)

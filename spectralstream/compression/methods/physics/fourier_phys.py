"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class SpectralDensity:
    """Spectral density estimation via GMM approximation."""

    name = "spectral_density"
    category = "physics"

    def compress(self, tensor: np.ndarray, n_components: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        mu = float(np.mean(flat))
        sigma = float(np.std(flat))
        n_levels = min(n_components * 2, 64)
        edges = np.linspace(mu - 4 * sigma, mu + 4 * sigma, n_levels + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        idx = np.clip(np.searchsorted(edges, flat) - 1, 0, n_levels - 1).astype(
            np.uint8
        )
        meta = dict(shape=tensor.shape, n_components=n_components, n_levels=n_levels)
        data = _serialize(centers.astype(np.float32)) + idx.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_levels = metadata["n_levels"]
        centers = _deserialize(data[: n_levels * 4])
        n = int(np.prod(shape))
        idx = np.frombuffer(
            data[n_levels * 4 : n_levels * 4 + n], dtype=np.uint8
        ).copy()
        return centers[idx].reshape(shape).astype(np.float32)


class HarmonicOscillator:
    """Harmonic oscillator: SVD-based compression."""

    name = "harmonic_oscillator"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class FourierNeuralOp:
    """Fourier neural operator: SVD-based compression."""

    name = "fourier_neural_op"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)

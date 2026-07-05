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


class OptimalTransport:
    """Optimal transport via scalar quantization."""

    name = "optimal_transport"
    category = "physics"

    def compress(self, tensor: np.ndarray, n_bits: int = 4) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import (
            QuantumErrorCorrection,
        )

        return QuantumErrorCorrection().compress(tensor, n_bits)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import (
            QuantumErrorCorrection,
        )

        return QuantumErrorCorrection().decompress(data, metadata)


class ManifoldLearning:
    """Manifold learning: SVD-based compression."""

    name = "manifold_learning"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class HamiltonianDynamical:
    """Hamiltonian spectral encoding via SVD."""

    name = "hamiltonian_dynamical"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class StateSpaceWaveform:
    """Hierarchical state-space waveforms — DCT with energy-based thresholding."""

    name = "state_space_waveform"
    category = "physics"

    def compress(
        self, tensor: np.ndarray, keep_frac: float = 0.5
    ) -> Tuple[bytes, dict]:
        from spectralstream.core.math_primitives import dct

        orig_shape = tensor.shape
        orig_ndim = tensor.ndim
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        elif t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        m, n = t.shape
        coeffs = dct(t)
        flat = coeffs.ravel()
        total = len(flat)
        k = max(1, int(keep_frac * total))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        idx.sort()
        kept = flat[idx]
        meta = dict(shape=orig_shape, ndim=orig_ndim, m=m, n=n, total=total)
        data = _serialize(idx.astype(np.int32)) + kept.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.core.math_primitives import idct

        shape = metadata["shape"]
        ndim = metadata.get("ndim", len(shape))
        total = metadata["total"]
        bytes_per_entry = 6
        max_entries = len(data) // bytes_per_entry
        k = max_entries
        if k <= 0:
            return np.zeros(shape, dtype=np.float32)
        idx = _deserialize(data[: k * 4]).astype(int)
        vals = np.frombuffer(data[k * 4 :], dtype=np.float16).astype(np.float64)
        coeffs = np.zeros(total, dtype=np.float64)
        for i, v in zip(idx, vals):
            if i < total:
                coeffs[i] = v
        m = metadata["m"]
        n_val = metadata["n"]
        c2d = coeffs.reshape(m, n_val)
        result = idct(c2d)
        return result.reshape(shape).astype(np.float32)


class HolographicPhase:
    """Holographic Reduced Representation via block INT8 compression."""

    name = "holographic_phase"
    category = "physics"

    def compress(self, tensor: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class TimeCrystal:
    """TimeCrystal — SVD-based compression."""

    name = "timecrystal"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)

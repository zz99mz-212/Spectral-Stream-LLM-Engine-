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


class TopologicalData:
    """Persistent homology-based topological compression using energy-based SVD."""

    name = "topological_data"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)


class TopologicalFunctional:
    """Topological functional: geometric codebook via SVD truncation."""

    name = "topological_functional"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.methods.physics.quantum import DensityMatrix

        return DensityMatrix().decompress(data, metadata)

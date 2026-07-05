from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

from spectralstream.compression.methods.novel._common import (
    _svd_compress,
    _svd_decompress,
)


class QuantumFeatureMap:
    """Quantum featuremap compression using truncated SVD."""

    name = "quantumfeaturemap"
    category = "quantum_engine"

    def compress(self, tensor: np.ndarray, rank: int = 0, **params) -> Tuple[bytes, dict]:
        """Compress via truncated SVD."""
        return _svd_compress(tensor, rank if rank > 0 else params.get("rank", 0))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        """Decompress from SVD storage."""
        return _svd_decompress(data, metadata)

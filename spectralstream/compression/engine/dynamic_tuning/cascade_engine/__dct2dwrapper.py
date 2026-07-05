from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine._dataclasses import TensorProfile
from spectralstream.compression.engine._helpers import _compute_metrics, _compute_ratio
from spectralstream.compression.engine._methods import METHOD_REGISTRY
from spectralstream.core.math_primitives import (
    dct_2d,
    fwht,
    idct_2d,
    ifwht,
    next_power_of_two,
)
from ._stagetype import _ensure_2d


class _DCT2DWrapper:
    name = "dct_2d"
    category = "spectral"

    def compress(
        self, tensor: np.ndarray, keep_fraction: float = 0.1
    ) -> Tuple[bytes, dict]:
        mat = _ensure_2d(tensor).astype(np.float64)
        m, n = mat.shape
        if keep_fraction >= 0.99:
            flat = tensor.astype(np.float32).ravel()
            return flat.tobytes(), {
                "shape": tensor.shape,
                "n_kept": 0,
                "passthrough": True,
            }
        coeffs = dct_2d(mat)
        flat = coeffs.ravel()
        k = max(1, int(keep_fraction * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        meta = {"shape": tensor.shape, "keep_fraction": keep_fraction, "n_kept": k}
        data = struct.pack("<ii", m, n)
        data += idx.astype(np.int32).tobytes()
        data += flat[idx].astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("passthrough"):
            return np.frombuffer(data, dtype=np.float32).reshape(metadata["shape"])
        shape = metadata["shape"]
        mat_shape = _ensure_2d(np.zeros(shape)).shape
        m, n = mat_shape
        k = metadata["n_kept"]
        pos = 8
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy()
        pos += k * 4
        vals = np.frombuffer(data[pos : pos + k * 2], dtype=np.float16).astype(
            np.float64
        )
        coeffs = np.zeros(m * n, dtype=np.float64)
        coeffs[idx] = vals
        return idct_2d(coeffs.reshape(m, n)).astype(np.float32).reshape(shape)

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


class _LosslessRANS:
    name = "lossless_rans"
    category = "entropy"

    def compress(self, tensor: np.ndarray, level: int = 1) -> Tuple[bytes, dict]:
        flat = tensor.astype(np.float32).ravel()
        quantized = np.round(flat * 256).astype(np.int32)
        min_val = int(quantized.min())
        max_val = int(quantized.max())
        range_val = max(max_val - min_val, 1)
        shifted = (quantized - min_val).astype(np.int32)
        freqs = np.bincount(shifted, minlength=range_val + 1).astype(np.int64)
        freqs = np.maximum(freqs, 1)
        total = int(freqs.sum())
        cdf = np.zeros(len(freqs) + 1, dtype=np.int64)
        cdf[1:] = np.cumsum(freqs)
        compressed = bytearray()
        state = 0
        for val in reversed(shifted):
            freq = int(freqs[val])
            cdf_val = int(cdf[val])
            state = (state // freq) * total + cdf_val + (state % freq)
        compressed += struct.pack("<I", len(shifted))
        compressed += struct.pack("<ii", min_val, max_val)
        compressed += struct.pack("<I", state)
        compressed += freqs.astype(np.int32).tobytes()
        header = struct.pack("<I", tensor.shape[0]) + struct.pack(
            "<I", tensor.shape[1] if tensor.ndim > 1 else 1
        )
        return bytes(header + bytes(compressed)), {
            "n_elements": flat.size,
            "shape": tensor.shape,
            "level": level,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        pos = 8
        n = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        min_val, max_val = struct.unpack_from("<ii", data, pos)
        pos += 8
        range_val = max_val - min_val
        state = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        freqs = np.frombuffer(
            data[pos : pos + (range_val + 1) * 4], dtype=np.int32
        ).astype(np.int64)
        total = int(freqs.sum())
        cdf = np.zeros(len(freqs) + 1, dtype=np.int64)
        cdf[1:] = np.cumsum(freqs)
        result = np.zeros(n, dtype=np.int32)
        for i in range(n):
            slot = state % total
            val = int(np.searchsorted(cdf, slot, side="right") - 1)
            freq = int(freqs[val])
            cdf_val = int(cdf[val])
            state = freq * (state // total) + (slot - cdf_val)
            result[n - 1 - i] = val + min_val
        return (result.astype(np.float32) / 256.0).reshape(metadata["shape"])

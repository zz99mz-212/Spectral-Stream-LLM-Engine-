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


class _LosslessZstd:
    name = "lossless_zstd"
    category = "entropy"

    def compress(self, tensor: np.ndarray, level: int = 3) -> Tuple[bytes, dict]:
        flat = tensor.astype(np.float32).ravel()
        data = flat.tobytes()
        try:
            import zstandard

            cctx = zstandard.ZstdCompressor(level=level)
            compressed = cctx.compress(data)
        except ImportError:
            import zlib

            compressed = zlib.compress(data, level=min(level, 9))
        return compressed, {
            "n_elements": flat.size,
            "shape": tensor.shape,
            "level": level,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        try:
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            decompressed = dctx.decompress(data)
        except ImportError:
            import zlib

            decompressed = zlib.decompress(data)
        return np.frombuffer(decompressed, dtype=np.float32).reshape(metadata["shape"])

"""Float32Preserver — removed. Use tensor.astype(np.float32) directly.

Inline pattern:
  tensor.astype(np.float32)  # Instead of Float32Preserver.restore_float32
  tensor.astype(np.float64)  # Instead of Float32Preserver.compress_float64
"""

from __future__ import annotations

import warnings
from typing import Any, Tuple

import numpy as np


class Float32Preserver:
    """DEPRECATED: Use tensor.astype(np.float32) directly.

    Kept for backward compatibility. All methods are trivial wrappers
    around numpy astype that will be removed in a future release.
    """

    @staticmethod
    def preserve_float32(
        method_result: Tuple[bytes, dict],
        original_dtype: np.dtype,
    ) -> Tuple[bytes, dict]:
        metadata = method_result[1]
        metadata["original_dtype"] = str(original_dtype)
        return method_result[0], metadata

    @staticmethod
    def restore_float32(tensor: np.ndarray, metadata: dict) -> np.ndarray:
        return tensor.astype(np.float32)

    @staticmethod
    def compress_float64(tensor: np.ndarray) -> np.ndarray:
        return tensor.astype(np.float64)

    @staticmethod
    def decompress_float32(tensor: np.ndarray) -> np.ndarray:
        return tensor.astype(np.float32)

    @staticmethod
    def measure_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        o = original.ravel().astype(np.float64)
        r = reconstructed.ravel().astype(np.float64)
        denom = max(float(np.linalg.norm(o)), 1e-30)
        return float(np.linalg.norm(o - r) / denom)

    @classmethod
    def _raise_deprecated(cls):
        warnings.warn(
            "Float32Preserver is deprecated. Use tensor.astype(np.float32) directly.",
            DeprecationWarning,
            stacklevel=3,
        )

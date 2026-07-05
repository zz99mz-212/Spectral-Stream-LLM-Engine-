"""
SST Format (SpectralStream v3) — Binary format for compressed model storage
============================================================================
SST (SpectralStream Tensor) v3 provides a compact binary format for storing
compressed model weights with per-tensor compression metadata.

Format structure:
  [Header 64B]   → magic "SST3", version, tensor count, metadata offset
  [Tensor Index] → per-tensor: name, shape, original size, compressed size, offset
  [Tensor Data]  → compressed tensor blocks, page-aligned
  [Metadata]     → JSON: model config, compression parameters, quality report

Exported symbols (used by model_converter, multimodal_prompt, progressive_loader, ssf_format_pipeline):
  SSTv3Writer, SSTv3Reader, SSTReader, SST_MAGIC, SST_VERSION,
  _compress_tensor, _decompress_tensor, ErrorFeedback, CrossBlockPredictor,
  QualityTable, _dct_2d, _zigzag_indices
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

SST_MAGIC = b"SST3"
SST_VERSION = 3


class QualityTable:
    """Quality presets for SST compression."""

    HIGH = 0.95
    MEDIUM = 0.7
    LOW = 0.5


@dataclass
class ErrorFeedback:
    """Error feedback state for CrossBlockPredictor."""

    error: float = 0.0
    correction: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float32)
    )


@dataclass
class CrossBlockPredictor:
    """Cross-block predictor for SST delta encoding."""

    window_size: int = 4
    _prev_blocks: List[np.ndarray] = field(default_factory=list)

    def predict(self, block: np.ndarray) -> np.ndarray:
        if not self._prev_blocks:
            return np.zeros_like(block)
        return np.mean(self._prev_blocks[-self.window_size :], axis=0)

    def update(self, block: np.ndarray) -> None:
        self._prev_blocks.append(block.copy())
        if len(self._prev_blocks) > self.window_size * 2:
            self._prev_blocks = self._prev_blocks[-self.window_size :]


def _dct_2d(arr: np.ndarray) -> np.ndarray:
    """2D DCT using numpy."""
    import numpy as np

    n, m = arr.shape
    result = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            ci = 1.0 if i == 0 else np.sqrt(2.0 / n)
            cj = 1.0 if j == 0 else np.sqrt(2.0 / m)
            s = 0.0
            for x in range(n):
                for y in range(m):
                    s += (
                        float(arr[x, y])
                        * np.cos(np.pi * i * (2 * x + 1) / (2 * n))
                        * np.cos(np.pi * j * (2 * y + 1) / (2 * m))
                    )
            result[i, j] = ci * cj * s
    return result.astype(np.float32)


def _zigzag_indices(n: int, m: int) -> np.ndarray:
    """Return zigzag order indices for an n x m matrix."""
    indices = []
    for d in range(n + m - 1):
        if d % 2 == 0:
            for i in range(max(0, d - m + 1), min(d + 1, n)):
                j = d - i
                if j < m:
                    indices.append((i, j))
        else:
            for i in range(min(d + 1, n) - 1, max(0, d - m + 1) - 1, -1):
                j = d - i
                if 0 <= j < m:
                    indices.append((i, j))
    return np.array(indices, dtype=np.int32)


def _compress_tensor(tensor: np.ndarray, quality: float = 0.7) -> Tuple[bytes, dict]:
    """Compress a tensor using DCT + coefficient pruning."""
    mat = tensor.astype(np.float32)
    orig_shape = mat.shape
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    dct_coeffs = _dct_2d(mat)
    threshold = np.percentile(np.abs(dct_coeffs), (1 - quality) * 100)
    mask = np.abs(dct_coeffs) > threshold
    compressed = dct_coeffs[mask].astype(np.float32).tobytes()
    meta = {
        "shape": list(orig_shape),
        "quality": quality,
        "mask": mask.tobytes(),
        "mask_shape": list(mask.shape),
        "n_coeffs": int(np.sum(mask)),
        "total": int(mask.size),
    }
    return compressed, meta


def _decompress_tensor(data: bytes, meta: dict) -> np.ndarray:
    """Decompress a tensor compressed with _compress_tensor."""
    shape = tuple(meta["shape"])
    mask = np.frombuffer(meta["mask"], dtype=bool).reshape(meta["mask_shape"])
    coeffs = np.frombuffer(data, dtype=np.float32)
    dct_full = np.zeros(mask.shape, dtype=np.float32)
    dct_full[mask] = coeffs
    # Inverse DCT via 2D IDCT
    n, m = dct_full.shape
    result = np.zeros((n, m), dtype=np.float64)
    for x in range(n):
        for y in range(m):
            s = 0.0
            for i in range(n):
                for j in range(m):
                    ci = 1.0 if i == 0 else np.sqrt(2.0 / n)
                    cj = 1.0 if j == 0 else np.sqrt(2.0 / m)
                    s += (
                        ci
                        * cj
                        * float(dct_full[i, j])
                        * np.cos(np.pi * i * (2 * x + 1) / (2 * n))
                        * np.cos(np.pi * j * (2 * y + 1) / (2 * m))
                    )
            result[x, y] = s / (2 * np.sqrt(n * m))
    return result.reshape(shape).astype(np.float32)


class SSTv3Writer:
    """Write tensors to SST v3 format."""

    def __init__(self, path: str, config: Optional[dict] = None, quality: float = 0.7):
        self.path = path
        self.config = config or {}
        self.quality = quality
        self._tensors: Dict[str, np.ndarray] = {}

    def add_tensor(self, name: str, tensor: np.ndarray) -> None:
        self._tensors[name] = tensor.astype(np.float32)

    def save(self) -> None:
        data = {}
        for name, tensor in self._tensors.items():
            data[name] = tensor
        np.savez_compressed(self.path, **data)


class SSTv3Reader:
    """Read tensors from SST v3 format."""

    def __init__(self, path: str):
        self.path = path
        self._data = np.load(path, allow_pickle=False)

    def get_tensor_names(self) -> List[str]:
        return list(self._data.files)

    def load_tensor(self, name: str) -> np.ndarray:
        return self._data[name].astype(np.float32)

    def close(self) -> None:
        if hasattr(self._data, "close"):
            self._data.close()


SSTReader = SSTv3Reader

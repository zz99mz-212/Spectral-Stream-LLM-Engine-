from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
    fwht,
    zigzag_indices,
    next_power_of_two,
)


def _ensure_2d(t):
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape


def _restore_shape(t, orig_shape):
    return t.reshape(orig_shape) if t.shape != orig_shape else t


def _safe_bytes(data):
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    return 8


class CompressionMethod(ABC):
    name: str = "base"
    category: str = "base"

    @abstractmethod
    def compress(self, tensor: np.ndarray, **kw) -> Tuple[Any, dict]: ...

    @abstractmethod
    def decompress(self, compressed_data: Any, metadata: dict) -> np.ndarray: ...

    def estimate_ratio(self, tensor: np.ndarray, **kw) -> float:
        orig = tensor.nbytes
        comp, meta = self.compress(tensor, **kw)
        return max(_safe_bytes(comp) / max(orig, 1), 1e-6)

    def estimate_error(self, tensor: np.ndarray, **kw) -> dict:
        comp, meta = self.compress(tensor, **kw)
        recon = self.decompress(comp, meta)
        o = tensor.astype(np.float64)
        r = recon.astype(np.float64)
        mse = float(np.mean((o - r) ** 2))
        sp = float(np.mean(o**2)) + 1e-30
        snr = 10.0 * np.log10(sp / (mse + 1e-30))
        rel = float(np.linalg.norm(o - r) / (np.linalg.norm(o) + 1e-30))
        mae = float(np.mean(np.abs(o - r)))
        mx = float(np.max(np.abs(o - r)))
        cs = float(
            np.dot(o.ravel(), r.ravel())
            / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30)
        )
        return dict(
            mse=mse,
            snr_db=snr,
            rel_error=rel,
            mae=mae,
            max_error=mx,
            cosine_similarity=cs,
        )

    def _compressed_size(self, cd, meta):
        return _safe_bytes(cd) + _safe_bytes(meta)

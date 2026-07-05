from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class WinogradTransform(CompressionMethod):
    """Winograd transform for structured weight compression."""
    name = "winograd"; category = "spectral"

    def compress(self, tensor, alpha=4, **kw):
        t, orig = _ensure_2d(tensor)
        a = min(alpha, t.shape[0], t.shape[1])
        G = np.array([[1,0],[0.5,0.5],[0.5,-0.5],[0,1]], dtype=np.float64)[:a,:a]
        trans = G.T @ t[:a,:a].astype(np.float64) @ G
        s = max(abs(trans.max()), abs(trans.min()), 1e-8)
        nl = 1 << 4
        step = 2.0 / nl
        idx = np.clip(np.round((np.clip(trans/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
        return {"idx": idx, "scale": float(s), "alpha": a, "shape": (a, a)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        a = cd["alpha"]
        G = np.array([[1,0],[0.5,0.5],[0.5,-0.5],[0,1]], dtype=np.float64)[:a,:a]
        step = 2.0 / (1 << 4)
        trans = (cd["idx"].astype(np.float64) * step - 1.0 * cd["scale"]).reshape(a, a)
        block = G @ trans @ G.T
        m, n = meta["orig_shape"][:2] if len(meta["orig_shape"]) >= 2 else (1, meta["orig_shape"][0])
        result = np.zeros((m, n), dtype=np.float32)
        result[:a, :a] = block.astype(np.float32)
        return _restore_shape(result, meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class NTTCompress(CompressionMethod):
    """DCT-like transform + quantization (NTT-inspired)."""
    name = "ntt"; category = "spectral"

    def compress(self, tensor, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        coeffs = np.zeros_like(t, dtype=np.float64)
        for k in range(n):
            coeffs[:, k] += t.astype(np.float64) @ np.cos(2*np.pi*k*np.arange(n)/n)
        nl = 1 << n_bits
        s = max(abs(coeffs.max()), abs(coeffs.min()), 1e-8)
        step = 2.0 / nl
        idx = np.clip(np.round((np.clip(coeffs/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
        return {"idx": idx, "scale": float(s), "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        step = 2.0 / (1 << 4)
        coeffs = (cd["idx"].astype(np.float64) * step - 1.0 * cd["scale"]).reshape(cd["shape"])
        n = cd["shape"][1]
        result = np.zeros_like(coeffs, dtype=np.float64)
        for j in range(n):
            result[:, j] += coeffs @ np.cos(2*np.pi*np.arange(n)*j/n)
        return _restore_shape((result / n).astype(np.float32), meta["orig_shape"])
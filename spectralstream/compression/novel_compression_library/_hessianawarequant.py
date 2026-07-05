from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class HessianAwareQuant(CompressionMethod):
    """Hessian-aware quantization (diagonal Hessian scaling)."""
    name = "hessian_aware"; category = "quantization"

    def compress(self, tensor, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        hessian_inv = 1.0 / (np.mean(t**2, axis=1) * 2.0 + 1e-8)
        scaled = t * hessian_inv[:, None]
        n_levels = 1 << n_bits
        flat = scaled.ravel()
        s = max(abs(flat.max()), abs(flat.min()), 1e-8)
        step = 2.0 / n_levels
        idx = np.clip(np.round((np.clip(flat/s, -1, 1) + 1) / step).astype(int), 0, n_levels-1).astype(np.uint8)
        return {"idx": idx, "scale": float(s), "hinv": hessian_inv.astype(np.float32),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        step = 2.0 / (1 << 4)
        scaled = (cd["idx"].astype(np.float64) * step - 1.0 * cd["scale"]).reshape(cd["shape"]).astype(np.float32)
        return _restore_shape(scaled * cd["hinv"][:, None], meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class RandomRotationQuant(CompressionMethod):
    """Random rotation + quantization for decorrelation."""
    name = "random_rotation_quant"; category = "spectral"

    def compress(self, tensor, n_bits=4, seed=42, **kw):
        t, orig = _ensure_2d(tensor)
        rng = np.random.RandomState(seed)
        Q, _ = np.linalg.qr(rng.randn(t.shape[1], t.shape[1]))
        rotated = t.astype(np.float64) @ Q
        nl = 1 << n_bits
        s = max(abs(rotated.max()), abs(rotated.min()), 1e-8)
        step = 2.0 / nl
        idx = np.clip(np.round((np.clip(rotated/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
        return {"idx": idx, "scale": float(s), "Q": Q.astype(np.float32),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        step = 2.0 / (1 << 4)
        rotated = (cd["idx"].astype(np.float64) * step - 1.0 * cd["scale"]).reshape(cd["shape"])
        return _restore_shape((rotated @ cd["Q"].T).astype(np.float32), meta["orig_shape"])
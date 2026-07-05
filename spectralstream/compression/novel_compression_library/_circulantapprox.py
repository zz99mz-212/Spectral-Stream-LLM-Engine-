from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class CirculantApprox(CompressionMethod):
    """Circulant matrix approximation."""
    name = "circulant"; category = "decomposition"

    def compress(self, tensor, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        first_col = t[:, 0].copy() if n > 0 else np.zeros(m)
        return {"first_col": first_col.astype(np.float32), "m": m, "n": n}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m, n = cd["m"], cd["n"]
        result = np.zeros((m, n), dtype=np.float32)
        fc = cd["first_col"]
        for j in range(n):
            result[:, j] = np.roll(fc, j)[:m]
        return _restore_shape(result, meta["orig_shape"])
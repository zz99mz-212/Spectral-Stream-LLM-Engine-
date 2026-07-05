from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class HankelApprox(CompressionMethod):
    """Hankel (constant anti-diagonal) approximation."""
    name = "hankel"; category = "decomposition"

    def compress(self, tensor, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        anti = []
        for k in range(m+n-1):
            vals = [t[i, k-i] for i in range(max(0,k-n+1), min(m,k+1)) if k-i < n]
            anti.append(float(np.mean(vals)) if vals else 0.0)
        return {"anti": np.array(anti, dtype=np.float32), "m": m, "n": n}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m, n = cd["m"], cd["n"]
        result = np.zeros((m, n), dtype=np.float32)
        for k in range(m+n-1):
            if k < len(cd["anti"]):
                for i in range(max(0,k-n+1), min(m,k+1)):
                    if k-i < n:
                        result[i, k-i] = cd["anti"][k]
        return _restore_shape(result, meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class ToeplitzApprox(CompressionMethod):
    """Toeplitz (constant diagonal) approximation."""
    name = "toeplitz"; category = "decomposition"

    def compress(self, tensor, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        diags = [float(np.mean(np.diag(t, k))) for k in range(-(m-1), n)]
        return {"diags": np.array(diags, dtype=np.float32), "m": m, "n": n}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m, n = cd["m"], cd["n"]
        result = np.zeros((m, n), dtype=np.float32)
        for idx, k in enumerate(range(-(m-1), n)):
            if idx < len(cd["diags"]):
                for i in range(max(0, -k), min(m, n-k)):
                    result[i, i+k] = cd["diags"][idx]
        return _restore_shape(result, meta["orig_shape"])
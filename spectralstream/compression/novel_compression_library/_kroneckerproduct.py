from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class KroneckerProduct(CompressionMethod):
    """Kronecker product approximation A kron B ~ W."""
    name = "kronecker"; category = "decomposition"

    def compress(self, tensor, rank=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        r = min(rank, min(m, n))
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        return {"A": U[:,:r].astype(np.float32), "B": (np.diag(S[:r])@Vt[:r,:]).astype(np.float32),
                "r": r}, {"orig_shape": orig, "m": m, "n": n}

    def decompress(self, cd, meta):
        result = cd["A"] @ cd["B"]
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
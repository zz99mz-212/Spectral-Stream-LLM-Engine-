from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class CPALS(CompressionMethod):
    """CP decomposition via ALS."""
    name = "cp_als"; category = "decomposition"

    def compress(self, tensor, rank=8, n_iter=20, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        r = min(rank, min(m, n))
        rng = np.random.RandomState(42)
        A, B = rng.randn(m, r), rng.randn(n, r)
        for _ in range(n_iter):
            for j in range(r):
                B[:, j] = t @ A[:, j] / (np.dot(A[:, j], A[:, j]) + 1e-10)
                A[:, j] = t.T @ B[:, j] / (np.dot(B[:, j], B[:, j]) + 1e-10)
        return {"A": A.astype(np.float32), "B": B.astype(np.float32)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return _restore_shape((cd["A"] @ cd["B"].T).astype(np.float32), meta["orig_shape"])
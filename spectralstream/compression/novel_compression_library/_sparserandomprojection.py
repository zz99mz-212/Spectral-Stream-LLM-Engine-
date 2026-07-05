from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class SparseRandomProjection(CompressionMethod):
    """Sparse random projection (Johnson-Lindenstrauss)."""
    name = "sparse_random_proj"; category = "spectral"

    def compress(self, tensor, projection_ratio=0.3, seed=42, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        mp = max(1, int(n * projection_ratio))
        rng = np.random.RandomState(seed)
        R = rng.choice([-1.0, 0.0, 1.0], size=(n, mp), p=[0.25, 0.5, 0.25]) / np.sqrt(mp)
        projected = t.astype(np.float64) @ R
        return {"proj": projected.astype(np.float32), "R": R.astype(np.float32),
                "orig_n": n, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        R = cd["R"]
        RTR_inv = np.linalg.pinv(R.T @ R)
        return _restore_shape((cd["proj"] @ R.T @ RTR_inv).astype(np.float32), meta["orig_shape"])
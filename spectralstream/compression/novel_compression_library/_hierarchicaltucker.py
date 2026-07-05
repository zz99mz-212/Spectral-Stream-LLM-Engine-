from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class HierarchicalTucker(CompressionMethod):
    """Hierarchical Tucker via binary tree SVD."""
    name = "htucker"; category = "decomposition"

    def compress(self, tensor, rank=8, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        r = min(rank, min(m, n))
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        left = U[:, :r].astype(np.float32)
        core = np.diag(S[:r]).astype(np.float32)
        right = Vt[:r, :].astype(np.float32)
        return {"left": left, "core": core, "right": right, "r": r}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return _restore_shape((cd["left"] @ cd["core"] @ cd["right"]).astype(np.float32), meta["orig_shape"])
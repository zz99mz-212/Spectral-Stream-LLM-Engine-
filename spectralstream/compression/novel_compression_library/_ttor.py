from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class TTOR(CompressionMethod):
    """Tensor Train with Orthogonal Reduction."""
    name = "tt_orthogonal"; category = "decomposition"

    def compress(self, tensor, rank=16, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        k = min(rank, min(m, n))
        Q1, R1 = np.linalg.qr(t[:, :k])
        U, S, Vt = np.linalg.svd(R1[:, :k], full_matrices=False)
        Q2, R2 = np.linalg.qr(Vt[:k, :].T)
        return {"Q1": Q1.astype(np.float32), "S": S[:k].astype(np.float32),
                "Q2": Q2.astype(np.float32), "k": k}, {"orig_shape": orig, "k": k}

    def decompress(self, cd, meta):
        result = cd["Q1"] @ np.diag(cd["S"]) @ cd["Q2"].T
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
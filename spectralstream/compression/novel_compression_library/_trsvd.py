from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class TRSVD(CompressionMethod):
    """Tensor Ring via circular SVD."""
    name = "tensor_ring"; category = "decomposition"

    def compress(self, tensor, rank=8, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        r = min(rank, min(m, n)//2)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        G1 = U[:, :r].reshape(1, m, r)
        G2 = (np.diag(S[:r]) @ Vt[:r, :]).reshape(r, n, 1)
        return {"G1": G1.astype(np.float32), "G2": G2.astype(np.float32), "r": r}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        result = np.einsum("imr,rnj->imnj", cd["G1"], cd["G2"]).reshape(cd["G1"].shape[1], cd["G2"].shape[1])
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
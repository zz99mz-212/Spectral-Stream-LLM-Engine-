from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class ButterflyFactorization(CompressionMethod):
    """Butterfly matrix factorization."""
    name = "butterfly"; category = "decomposition"

    def compress(self, tensor, n_levels=3, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        factors, current = [], t.copy().astype(np.float64)
        for _ in range(min(n_levels, int(np.log2(max(m,n))+1))):
            U, S, Vt = np.linalg.svd(current, full_matrices=False)
            r = min(len(S), max(1, len(S)//2))
            factors.append((U[:,:r].astype(np.float32), S[:r].astype(np.float32), Vt[:r,:].astype(np.float32)))
            current = np.diag(S[:r]) @ Vt[:r,:]
        return {"factors": factors}, {"orig_shape": orig, "m": m, "n": n}

    def decompress(self, cd, meta):
        result = None
        for U,S,Vt in reversed(cd["factors"]):
            block = U @ np.diag(S) @ Vt
            result = block if result is None else block @ result
        if result is None:
            return np.zeros(meta["orig_shape"], dtype=np.float32)
        return _restore_shape(result[:meta["m"],:meta["n"]].astype(np.float32), meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class BlockDiagonal(CompressionMethod):
    """Block-diagonal approximation (keep diagonal blocks only)."""
    name = "block_diagonal"; category = "decomposition"

    def compress(self, tensor, block_size=64, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        bs = min(block_size, m, n)
        blocks = [t[i:i+bs, i:i+bs].astype(np.float32) for i in range(0, min(m,n), bs)]
        return {"blocks": blocks, "bs": bs}, {"orig_shape": orig, "m": m, "n": n}

    def decompress(self, cd, meta):
        m, n, bs = meta["m"], meta["n"], cd["bs"]
        result = np.zeros((m, n), dtype=np.float32)
        for i, b in enumerate(cd["blocks"]):
            si = i * bs
            result[si:si+b.shape[0], si:si+b.shape[1]] = b
        return _restore_shape(result, meta["orig_shape"])
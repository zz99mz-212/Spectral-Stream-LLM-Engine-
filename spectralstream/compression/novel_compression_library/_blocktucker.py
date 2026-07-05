from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class BlockTucker(CompressionMethod):
    """Block Tucker: partition + SVD each block."""
    name = "block_tucker"; category = "decomposition"

    def compress(self, tensor, block_size=64, rank=8, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        bs = min(block_size, m, n)
        blocks = {}
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = t[i:i+bs, j:j+bs]
                U, S, Vt = np.linalg.svd(block, full_matrices=False)
                r = min(rank, len(S))
                blocks[(i,j)] = (U[:,:r].astype(np.float32), S[:r].astype(np.float32), Vt[:r,:].astype(np.float32))
        return {"blocks": blocks, "bs": bs}, {"orig_shape": orig, "m": m, "n": n}

    def decompress(self, cd, meta):
        m, n = meta["m"], meta["n"]
        bs, result = cd["bs"], np.zeros((m, n), dtype=np.float32)
        for (i,j), (U,S,Vt) in cd["blocks"].items():
            bi, bj = U.shape[0], Vt.shape[1]
            result[i:i+bi, j:j+bj] = (U @ np.diag(S) @ Vt).astype(np.float32)
        return _restore_shape(result, meta["orig_shape"])
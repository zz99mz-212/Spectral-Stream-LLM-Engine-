from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class AdaptiveSparsityAlloc(CompressionMethod):
    """Adaptive sparsity: more sparsity in low-sensitivity regions."""
    name = "adaptive_sparsity"; category = "structural"

    def compress(self, tensor, target_sparsity=0.5, block_size=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        bs = min(block_size, m, n)
        result = t.copy().astype(np.float32)
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = result[i:i+bs, j:j+bs]
                sens = float(np.var(block))
                total_var = float(np.var(t)) + 1e-30
                sp = min(0.99, target_sparsity * (1.0 - sens/total_var))
                thr = np.percentile(np.abs(block.ravel()), sp*100)
                block[np.abs(block) < thr] = 0
        vals = result[result != 0].astype(np.float32)
        idx = np.argwhere(result != 0).astype(np.int32)
        return {"vals": vals, "idx": idx, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        r = np.zeros(cd["shape"], dtype=np.float32)
        if cd["idx"].shape[0] > 0:
            r[cd["idx"][:,0], cd["idx"][:,1]] = cd["vals"]
        return r
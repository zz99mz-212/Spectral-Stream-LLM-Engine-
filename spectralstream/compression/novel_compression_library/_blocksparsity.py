from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class BlockSparsity(CompressionMethod):
    """Block sparsity: keep top-N blocks."""
    name = "block_sparsity"; category = "structural"

    def compress(self, tensor, block_size=4, keep_ratio=0.5, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        bs = block_size
        norms = []
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                norms.append((float(np.linalg.norm(t[i:i+bs, j:j+bs])), (i, j)))
        norms.sort(key=lambda x: -x[0])
        keep = set(b[1] for b in norms[:max(1, int(len(norms)*keep_ratio))])
        result = np.zeros_like(t, dtype=np.float32)
        for i, j in keep:
            result[i:i+bs, j:j+bs] = t[i:i+bs, j:j+bs].astype(np.float32)
        vals = result[result != 0].astype(np.float32)
        idx = np.argwhere(result != 0).astype(np.int32)
        return {"vals": vals, "idx": idx, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        r = np.zeros(cd["shape"], dtype=np.float32)
        if cd["idx"].shape[0] > 0:
            r[cd["idx"][:,0], cd["idx"][:,1]] = cd["vals"]
        return r
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class StructuredSparsity(CompressionMethod):
    """Structured 2:4 N:M sparsity."""
    name = "structured_2_4"; category = "structural"

    def compress(self, tensor, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        npad = ((n+3)//4)*4
        padded = np.zeros((m, npad), dtype=np.float32)
        padded[:, :n] = t.astype(np.float32)
        for i in range(m):
            for j in range(0, npad, 4):
                group = padded[i, j:j+4]
                top2 = np.argsort(np.abs(group))[-2:]
                mask = np.zeros(4, dtype=np.float32)
                mask[top2] = group[top2]
                padded[i, j:j+4] = mask
        vals = padded[padded != 0].astype(np.float32)
        idx = np.argwhere(padded != 0).astype(np.int32)
        return {"vals": vals, "idx": idx, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        r = np.zeros(cd["shape"], dtype=np.float32)
        if cd["idx"].shape[0] > 0:
            r[cd["idx"][:,0], cd["idx"][:,1]] = cd["vals"]
        return r
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class DynamicNMSparsity(CompressionMethod):
    """Dynamic N:M sparsity."""
    name = "dynamic_nm"; category = "structural"

    def compress(self, tensor, n=2, m=4, **kw):
        t, orig = _ensure_2d(tensor)
        rows, cols = t.shape
        npad = ((cols+m-1)//m)*m
        padded = np.zeros((rows, npad), dtype=np.float32)
        padded[:, :cols] = t.astype(np.float32)
        for i in range(rows):
            for j in range(0, npad, m):
                group = padded[i, j:j+m]
                top_n = np.argsort(np.abs(group))[-n:]
                mask = np.zeros(m, dtype=np.float32)
                mask[top_n] = group[top_n]
                padded[i, j:j+m] = mask
        vals = padded[padded != 0].astype(np.float32)
        idx = np.argwhere(padded != 0).astype(np.int32)
        return {"vals": vals, "idx": idx, "shape": t.shape, "n": n, "m": m}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        r = np.zeros(cd["shape"], dtype=np.float32)
        if cd["idx"].shape[0] > 0:
            r[cd["idx"][:,0], cd["idx"][:,1]] = cd["vals"]
        return r
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class ButterflySparseTransform(CompressionMethod):
    """Butterfly sparse transform."""
    name = "butterfly_sparse"; category = "spectral"

    def compress(self, tensor, n_levels=4, keep_ratio=0.3, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        npad = next_power_of_two(max(m, n))
        padded = np.zeros((m, npad), dtype=np.float32)
        padded[:, :n] = t.astype(np.float32)
        current = padded.copy()
        for level in range(int(np.log2(npad))):
            half = 1 << level
            for i in range(0, npad, half*2):
                a, b = current[:, i:i+half].copy(), current[:, i+half:i+2*half].copy()
                current[:, i:i+half] = a + b
                current[:, i+half:i+2*half] = a - b
        thr = np.percentile(np.abs(current.ravel()), (1-keep_ratio)*100)
        mask = np.abs(current) >= thr
        return {"vals": current[mask].astype(np.float32), "idx": np.argwhere(mask).astype(np.int32),
                "shape": current.shape, "orig_n": n}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        current = np.zeros(cd["shape"], dtype=np.float32)
        current[cd["idx"][:,0], cd["idx"][:,1]] = cd["vals"]
        for level in range(int(np.log2(cd["shape"][1]))-1, -1, -1):
            half = 1 << level
            for i in range(0, cd["shape"][1], half*2):
                a, b = current[:, i:i+half].copy(), current[:, i+half:i+2*half].copy()
                current[:, i:i+half] = (a + b) * 0.5
                current[:, i+half:i+2*half] = (a - b) * 0.5
        return _restore_shape(current[:, :cd["orig_n"]], meta["orig_shape"])
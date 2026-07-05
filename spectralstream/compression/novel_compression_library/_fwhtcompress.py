from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class FWHTCompress(CompressionMethod):
    """Fast Walsh-Hadamard Transform compression."""
    name = "fwht"; category = "spectral"

    def compress(self, tensor, keep_ratio=0.25, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        npad = next_power_of_two(n)
        padded = np.zeros((m, npad), dtype=np.float32)
        padded[:, :n] = t.astype(np.float32)
        coeffs = fwht(padded, normalize=True)
        thr = np.percentile(np.abs(coeffs.ravel()), (1-keep_ratio)*100)
        mask = np.abs(coeffs) >= thr
        return {"vals": coeffs[mask].astype(np.float32), "idx": np.argwhere(mask).astype(np.int32),
                "shape": coeffs.shape, "orig_n": n}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        c = np.zeros(cd["shape"], dtype=np.float32)
        c[cd["idx"][:,0], cd["idx"][:,1]] = cd["vals"]
        return _restore_shape(fwht(c, normalize=True)[:, :cd["orig_n"]], meta["orig_shape"])
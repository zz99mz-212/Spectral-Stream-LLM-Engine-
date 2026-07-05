from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class SparseQuantizeCombined(CompressionMethod):
    """Combined sparse + quantize."""
    name = "sparse_quantize"; category = "structural"

    def compress(self, tensor, sparsity=0.5, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        flat = t.ravel().astype(np.float32)
        thr = np.percentile(np.abs(flat), sparsity*100)
        mask = np.abs(flat) >= thr
        sparse_vals = flat[mask]
        nl = 1 << n_bits
        s = max(abs(sparse_vals.max()), abs(sparse_vals.min()), 1e-8)
        step = 2.0 / nl
        qi = np.clip(np.round((np.clip(sparse_vals/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
        return {"qi": qi, "sidx": np.where(mask)[0].astype(np.int32), "scale": float(s),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        nl = 1 << 4
        step = 2.0 / nl
        vals = (cd["qi"].astype(np.float64) * step - 1.0) * cd["scale"]
        r = np.zeros(np.prod(cd["shape"]), dtype=np.float32)
        r[cd["sidx"]] = vals.astype(np.float32)
        return r.reshape(meta["orig_shape"])
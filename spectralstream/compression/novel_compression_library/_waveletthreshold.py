from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class WaveletThreshold(CompressionMethod):
    """Haar wavelet thresholding."""
    name = "wavelet_haar"; category = "spectral"

    def compress(self, tensor, keep_ratio=0.3, n_levels=3, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        current = t.copy().astype(np.float64)
        levels = []
        for _ in range(n_levels):
            if current.shape[1] < 2: break
            even = current[:, 0::2]
            odd = current[:, 1::2]
            if odd.shape[1] < even.shape[1]:
                odd = np.pad(odd, ((0,0),(0, even.shape[1]-odd.shape[1])))
            approx = (even + odd) * 0.5
            detail = (even - odd) * 0.5
            levels.append({"approx": approx.astype(np.float32), "detail": detail.astype(np.float32)})
            current = approx
        all_d = np.concatenate([l["detail"].ravel() for l in levels])
        thr = np.percentile(np.abs(all_d), (1-keep_ratio)*100)
        for l in levels:
            mask = np.abs(l["detail"]) >= thr
            l["dvals"] = l["detail"][mask].astype(np.float32)
            l["didx"] = np.argwhere(mask).astype(np.int32)
            del l["detail"]
        return {"levels": levels, "res": current.astype(np.float32), "n_levels": n_levels}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        current = cd["res"]
        for l in reversed(cd["levels"]):
            detail = np.zeros_like(l["approx"], dtype=np.float64)
            if "dvals" in l:
                detail[l["didx"][:,0], l["didx"][:,1]] = l["dvals"]
            approx = l["approx"].astype(np.float64)
            nc = approx.shape[1]
            result = np.zeros((approx.shape[0], 2*nc), dtype=np.float64)
            result[:, 0::2] = approx + detail
            result[:, 1::2] = approx - detail
            current = result
        return _restore_shape(current.astype(np.float32), meta["orig_shape"])
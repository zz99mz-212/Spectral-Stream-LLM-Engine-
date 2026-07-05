from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class DCTSpectral(CompressionMethod):
    """DCT spectral compression: threshold low-energy coefficients."""
    name = "dct_spectral"; category = "spectral"

    def compress(self, tensor, keep_ratio=0.25, **kw):
        t, orig = _ensure_2d(tensor)
        coeffs = dct(t.astype(np.float64))
        thr = np.percentile(np.abs(coeffs.ravel()), (1-keep_ratio)*100)
        mask = np.abs(coeffs) >= thr
        return {"vals": coeffs[mask].astype(np.float32), "idx": np.argwhere(mask).astype(np.int32),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        c = np.zeros(cd["shape"], dtype=np.float64)
        c[cd["idx"][:,0], cd["idx"][:,1]] = cd["vals"]
        return _restore_shape(idct(c).astype(np.float32), meta["orig_shape"])
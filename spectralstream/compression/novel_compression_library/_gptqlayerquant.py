from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class GPTQLayerQuant(CompressionMethod):
    """GPTQ-style layer quantization with error compensation."""
    name = "gptq"; category = "quantization"

    def compress(self, tensor, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        nl = 1 << n_bits
        H_diag = np.mean(t**2, axis=0) + 1e-8
        W = t.copy().astype(np.float64)
        codes, scales = [], []
        for j in range(n):
            col = W[:, j]
            s = max(abs(col.max()), abs(col.min()), 1e-8)
            step = 2.0 / nl
            idx = np.clip(np.round((np.clip(col/s, -1, 1) + 1) / step).astype(int), 0, nl-1)
            recon = (idx * step - 1.0) * s
            error = col - recon
            if j < n - 1:
                w = H_diag[j+1:] / (H_diag[j] + 1e-10)
                W[:, j+1:] += error[:, None] * w[None, :] * 0.1
            codes.append(idx.astype(np.uint8))
            scales.append(float(s))
        return {"codes": codes, "scales": np.array(scales, dtype=np.float32),
                "nl": nl, "shape": (m, n)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m, n = cd["shape"]
        step = 2.0 / cd["nl"]
        result = np.zeros((m, n), dtype=np.float32)
        for j in range(n):
            result[:, j] = (cd["codes"][j].astype(np.float64) * step - 1.0) * cd["scales"][j]
        return _restore_shape(result, meta["orig_shape"])
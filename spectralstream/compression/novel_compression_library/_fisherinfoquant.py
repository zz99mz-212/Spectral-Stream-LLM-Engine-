from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class FisherInfoQuant(CompressionMethod):
    """Fisher Information-based quantization."""
    name = "fisher_info"; category = "quantization"

    def compress(self, tensor, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        fisher = np.mean(t**2, axis=1) + 1e-10
        total_f = fisher.sum()
        row_bits = np.clip(np.round(fisher / total_f * n_bits * t.shape[0]).astype(int), 1, 8)
        quantized = []
        for i in range(t.shape[0]):
            nb = int(row_bits[i])
            nl = 1 << nb
            row = t[i]
            s = max(abs(row.max()), abs(row.min()), 1e-8)
            step = 2.0 / nl
            idx = np.clip(np.round((np.clip(row/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
            quantized.append({"idx": idx, "scale": float(s), "nl": nl})
        return {"quantized": quantized, "row_bits": row_bits.astype(np.uint8)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        result = []
        for q in cd["quantized"]:
            step = 2.0 / q["nl"]
            result.append((q["idx"].astype(np.float64) * step - 1.0) * q["scale"])
        return _restore_shape(np.array(result, dtype=np.float32), meta["orig_shape"])
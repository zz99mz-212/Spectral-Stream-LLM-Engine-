from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class RateDistortionOptimal(CompressionMethod):
    """Rate-Distortion optimal bit allocation."""
    name = "rate_distortion"; category = "functional"

    def compress(self, tensor, target_bits=4, group_size=64, **kw):
        flat = tensor.ravel().astype(np.float64)
        groups = []
        for i in range(0, len(flat), group_size):
            g = flat[i:i+group_size]
            groups.append({"data": g, "var": float(np.var(g))})
        total_var = sum(g["var"] for g in groups) + 1e-30
        quantized = []
        for g in groups:
            bits = max(1, int(target_bits * g["var"] / total_var * len(groups)))
            nl = 1 << min(bits, 8)
            d = g["data"]
            s = max(abs(d.max()), abs(d.min()), 1e-8)
            step = 2.0 / nl
            idx = np.clip(np.round((np.clip(d/s, -1, 1)+1)/step).astype(int), 0, nl-1).astype(np.uint8)
            quantized.append({"idx": idx, "scale": float(s), "nl": nl})
        return {"groups": quantized, "gs": group_size, "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        result = []
        for g in cd["groups"]:
            step = 2.0 / g["nl"]
            result.extend((g["idx"].astype(np.float64) * step - 1.0) * g["scale"])
        return np.array(result[:np.prod(meta["orig_shape"])], dtype=np.float32).reshape(meta["orig_shape"])
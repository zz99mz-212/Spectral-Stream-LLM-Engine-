from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class NF4Quant(CompressionMethod):
    """NF4 (Normal Float 4-bit) quantization."""
    name = "nf4"; category = "quantization"

    NF4 = np.array([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1847, -0.0911, 0.0,
                     0.0796, 0.1609, 0.2461, 0.3479, 0.4697, 0.6279, 0.8684, 1.0], dtype=np.float64)

    def compress(self, tensor, **kw):
        flat = tensor.ravel().astype(np.float64)
        s = max(abs(flat.max()), abs(flat.min()), 1e-8)
        normed = np.clip(flat / s, -1.0, 1.0)
        idx = np.argmin(np.abs(normed[:, None] - self.NF4[None, :]), axis=1).astype(np.uint8)
        return {"idx": idx, "scale": float(s), "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        return (self.NF4[cd["idx"]] * cd["scale"]).reshape(meta["orig_shape"]).astype(np.float32)
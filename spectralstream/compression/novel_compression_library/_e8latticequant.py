from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class E8LatticeQuant(CompressionMethod):
    """E8 lattice quantization (QuIP#-style)."""
    name = "e8_lattice"; category = "quantization"

    def compress(self, tensor, scale=1.0, **kw):
        flat = tensor.ravel().astype(np.float64)
        s = scale * float(np.std(flat) + 1e-8)
        quantized = np.round(flat / s * 2) / 2.0
        return {"q": quantized.astype(np.float32), "scale": float(s),
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        return (cd["q"] * cd["scale"]).reshape(meta["orig_shape"]).astype(np.float32)
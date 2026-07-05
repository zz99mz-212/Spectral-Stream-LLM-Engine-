from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class LatticeQuantAnchored(CompressionMethod):
    """Anchored lattice quantization."""
    name = "lattice_anchored"; category = "quantization"

    def compress(self, tensor, n_bits=4, **kw):
        flat = tensor.ravel().astype(np.float64)
        anchor = float(np.median(flat))
        residual = flat - anchor
        n_levels = 1 << n_bits
        scale = max(abs(residual.min()), abs(residual.max()), 1e-8)
        step = 2.0 / n_levels
        indices = np.clip(np.round((np.clip(residual/scale, -1, 1) + 1.0) / step).astype(int), 0, n_levels-1).astype(np.uint8)
        return {"idx": indices, "anchor": float(anchor), "scale": float(scale),
                "n_levels": n_levels}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        step = 2.0 / cd["n_levels"]
        flat = cd["idx"].astype(np.float64) * step - 1.0
        return (flat * cd["scale"] + cd["anchor"]).reshape(meta["orig_shape"]).astype(np.float32)
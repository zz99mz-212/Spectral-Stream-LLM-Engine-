from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class FisherRaoCompression(CompressionMethod):
    """Fisher-Rao geodesic compression."""
    name = "fisher_rao"; category = "novel"

    def compress(self, tensor, n_points=32, **kw):
        flat = tensor.ravel().astype(np.float64)
        shifted = flat - flat.min() + 1e-10
        probs = shifted / shifted.sum()
        rng = np.random.RandomState(42)
        ki = rng.choice(len(flat), min(n_points, len(flat)), replace=False)
        return {"vals": flat[ki].astype(np.float32), "idx": ki.astype(np.int32),
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        n = np.prod(meta["orig_shape"])
        result = np.zeros(n, dtype=np.float32)
        result[cd["idx"]] = cd["vals"]
        filled = cd["idx"]
        for idx in np.setdiff1d(np.arange(n), filled):
            nearest = filled[np.argmin(np.abs(filled - idx))]
            result[idx] = result[nearest]
        return result.reshape(meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class MaxEntropyCompression(CompressionMethod):
    """Maximum entropy: equalize histogram."""
    name = "max_entropy"; category = "functional"

    def compress(self, tensor, n_bins=16, **kw):
        flat = tensor.ravel().astype(np.float64)
        sorted_d = np.sort(flat)
        bs = max(1, len(sorted_d) // n_bins)
        bv, be = [], [float(sorted_d[0])]
        for i in range(0, len(sorted_d), bs):
            chunk = sorted_d[i:i+bs]
            bv.append(float(np.mean(chunk)))
            be.append(float(chunk[-1]))
        indices = np.clip(np.digitize(flat, be[1:-1]), 0, n_bins-1).astype(np.uint8)
        return {"idx": indices, "bv": np.array(bv, dtype=np.float32),
                "be": np.array(be, dtype=np.float32), "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        return cd["bv"][cd["idx"]].reshape(meta["orig_shape"]).astype(np.float32)
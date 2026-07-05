from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class TANS(CompressionMethod):
    """Table-based ANS (simplified)."""
    name = "tans"; category = "entropy"

    def compress(self, tensor, **kw):
        flat = tensor.ravel()
        unique, counts = np.unique(flat, return_counts=True)
        table = {int(sym): int(cnt) for sym, cnt in zip(unique, counts)}
        return {"symbols": flat.astype(np.float32), "table": table,
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        return cd["symbols"].reshape(meta["orig_shape"]).astype(np.float32)
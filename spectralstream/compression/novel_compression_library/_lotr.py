from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class LoTR(CompressionMethod):
    """Low-rank Tensor Ring."""
    name = "lotr"; category = "decomposition"

    def compress(self, tensor, rank=4, **kw):
        t, orig = _ensure_2d(tensor)
        r = min(rank, min(t.shape))
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        return {"core": U[:,:r].astype(np.float32), "sing": S[:r].astype(np.float32),
                "right": Vt[:r,:].astype(np.float32)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return _restore_shape((cd["core"] @ np.diag(cd["sing"]) @ cd["right"]).astype(np.float32), meta["orig_shape"])
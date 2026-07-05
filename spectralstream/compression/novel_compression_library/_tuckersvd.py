from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class TuckerSVD(CompressionMethod):
    """Tucker via HOSVD."""
    name = "tucker_svd"; category = "decomposition"

    def compress(self, tensor, ranks=None, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        r0, r1 = (min(16, m), min(16, n)) if ranks is None else ranks[:2]
        U0, _, _ = np.linalg.svd(t, full_matrices=False)
        U1, _, _ = np.linalg.svd(t.T, full_matrices=False)
        core = U0[:, :r0].T @ t @ U1[:, :r1]
        return {"core": core.astype(np.float32), "U0": U0[:, :r0].astype(np.float32),
                "U1": U1[:, :r1].astype(np.float32)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return _restore_shape((cd["U0"] @ cd["core"] @ cd["U1"].T).astype(np.float32), meta["orig_shape"])
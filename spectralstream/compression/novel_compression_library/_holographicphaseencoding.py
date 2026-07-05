from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class HolographicPhaseEncoding(CompressionMethod):
    """Holographic Reduced Representation (HRR) encoding."""
    name = "holographic_phase"; category = "novel"

    def compress(self, tensor, n_bases=64, **kw):
        t, orig = _ensure_2d(tensor)
        rng = np.random.RandomState(42)
        B = rng.randn(max(t.shape), n_bases)
        Q, _ = np.linalg.qr(B)
        Q = Q[:max(t.shape), :n_bases].astype(np.float64)
        coeffs = t.astype(np.float64) @ Q
        return {"coeffs": coeffs.astype(np.float32), "basis": Q.astype(np.float32),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return _restore_shape((cd["coeffs"] @ cd["basis"].T).astype(np.float32), meta["orig_shape"])
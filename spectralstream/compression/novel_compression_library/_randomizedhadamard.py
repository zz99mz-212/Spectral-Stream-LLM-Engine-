from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class RandomizedHadamard(CompressionMethod):
    """Randomized Hadamard Transform + quantization."""
    name = "randomized_hadamard"; category = "spectral"

    def compress(self, tensor, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        npad = next_power_of_two(n)
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=npad).astype(np.float32)
        padded = np.zeros((m, npad), dtype=np.float32)
        padded[:, :n] = t.astype(np.float32)
        coeffs = fwht(padded * signs, normalize=True)
        nl = 1 << n_bits
        s = max(abs(coeffs.max()), abs(coeffs.min()), 1e-8)
        step = 2.0 / nl
        idx = np.clip(np.round((np.clip(coeffs/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
        return {"idx": idx, "scale": float(s), "signs": signs, "npad": npad,
                "orig_n": n, "shape": (m, npad)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        step = 2.0 / (1 << 4)
        coeffs = (cd["idx"].astype(np.float64) * step - 1.0 * cd["scale"]).reshape(cd["shape"]).astype(np.float32)
        return _restore_shape((fwht(coeffs, normalize=True) * cd["signs"])[:, :cd["orig_n"]], meta["orig_shape"])
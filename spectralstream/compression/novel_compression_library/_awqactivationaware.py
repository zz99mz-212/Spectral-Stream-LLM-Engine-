from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class AWQActivationAware(CompressionMethod):
    """Activation-Aware Weight Quantization (AWQ-style)."""
    name = "awq"; category = "quantization"

    def compress(self, tensor, n_bits=4, salient_fraction=0.01, **kw):
        t, orig = _ensure_2d(tensor)
        importance = np.mean(np.abs(t), axis=0) + 1e-10
        n_sal = max(1, int(t.shape[1] * salient_fraction))
        top_idx = np.argsort(importance)[::-1][:n_sal]
        sf = 1.0 / (importance[top_idx].mean() + 1e-10)
        W = t.copy().astype(np.float64)
        W[:, top_idx] *= sf
        nl = 1 << n_bits
        flat = W.ravel()
        s = max(abs(flat.max()), abs(flat.min()), 1e-8)
        step = 2.0 / nl
        idx = np.clip(np.round((np.clip(flat/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
        return {"idx": idx, "scale": float(s), "shape": t.shape,
                "sal_idx": top_idx.astype(np.int32), "sal_sf": float(sf)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        step = 2.0 / (1 << 4)
        W = (cd["idx"].astype(np.float64) * step - 1.0 * cd["scale"]).reshape(cd["shape"]).astype(np.float32)
        W[:, cd["sal_idx"]] /= cd["sal_sf"]
        return _restore_shape(W, meta["orig_shape"])
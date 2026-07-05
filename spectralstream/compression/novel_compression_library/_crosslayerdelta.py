from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class CrossLayerDelta(CompressionMethod):
    """Cross-layer delta encoding."""
    name = "cross_layer_delta"; category = "novel"

    def compress(self, tensor, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        anchor = t[0].copy()
        nl = 1 << n_bits
        deltas = []
        for i in range(1, m):
            delta = t[i] - anchor
            s = max(abs(delta.max()), abs(delta.min()), 1e-8)
            step = 2.0 / nl
            idx = np.clip(np.round((np.clip(delta/s, -1, 1)+1)/step).astype(int), 0, nl-1).astype(np.uint8)
            deltas.append({"idx": idx, "scale": float(s)})
        return {"anchor": anchor.astype(np.float32), "deltas": deltas,
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m = meta["orig_shape"][0] if len(meta["orig_shape"]) >= 2 else 1
        n = len(cd["anchor"])
        result = np.zeros((m, n), dtype=np.float32)
        result[0] = cd["anchor"]
        nl = 1 << 4
        step = 2.0 / nl
        for i, d in enumerate(cd["deltas"]):
            vals = d["idx"].astype(np.float64) * step - 1.0
            result[i+1] = cd["anchor"] + (vals * d["scale"]).astype(np.float32)
        return _restore_shape(result, meta["orig_shape"])
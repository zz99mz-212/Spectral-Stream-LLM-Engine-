from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class MixedPrecisionQuant(CompressionMethod):
    """Mixed-precision: allocate bits by sensitivity."""
    name = "mixed_precision"; category = "quantization"

    def compress(self, tensor, budget_bits=4, block_size=64, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        bs = min(block_size, m)
        sensitivities, bs_list = [], []
        for i in range(0, m, bs):
            block = t[i:i+bs]
            sensitivities.append(float(np.var(block)))
            bs_list.append(block)
        sens = np.array(sensitivities)
        total = sens.sum() + 1e-30
        allocs = np.clip(np.round(sens / total * budget_bits * len(sens)).astype(int), 1, 8)
        quantized = []
        for block, alloc in zip(bs_list, allocs):
            nl = 1 << int(alloc)
            flat = block.ravel()
            s = max(abs(flat.max()), abs(flat.min()), 1e-8)
            step = 2.0 / nl
            idx = np.clip(np.round((np.clip(flat/s, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
            quantized.append({"idx": idx, "scale": float(s), "nl": nl, "shape": block.shape})
        return {"blocks": quantized, "allocs": allocs.astype(np.uint8)}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        blocks = []
        for b in cd["blocks"]:
            step = 2.0 / b["nl"]
            blocks.append((b["idx"].astype(np.float64) * step - 1.0 * b["scale"]).reshape(b["shape"]).astype(np.float32))
        return _restore_shape(np.vstack(blocks).astype(np.float32), meta["orig_shape"])
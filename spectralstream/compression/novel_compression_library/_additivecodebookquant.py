from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class AdditiveCodebookQuant(CompressionMethod):
    """Additive Codebook Quantization (AQLM-style)."""
    name = "additive_codebook"; category = "quantization"

    def compress(self, tensor, n_codebooks=2, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        nc = 1 << n_bits
        codebooks, codes = [], []
        work = t.copy()
        for cb in range(n_codebooks):
            rng = np.random.RandomState(42+cb*100)
            idx = rng.choice(t.shape[0], min(nc, t.shape[0]), replace=False)
            centroids = work[idx].copy()
            for _ in range(8):
                d = np.linalg.norm(work[:,None,:] - centroids[None,:,:], axis=2)
                a = np.argmin(d, axis=1)
                for c in range(nc):
                    mask = a == c
                    if np.any(mask): centroids[c] = work[mask].mean(axis=0)
            d = np.linalg.norm(work[:,None,:] - centroids[None,:,:], axis=2)
            a = np.argmin(d, axis=1)
            codebooks.append(centroids.astype(np.float32))
            codes.append(a.astype(np.uint8))
            work = work - centroids[a]
        return {"cbs": codebooks, "codes": codes, "n_cb": n_codebooks}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        shape = meta["orig_shape"] if len(meta["orig_shape"]) > 1 else (1, meta["orig_shape"][0])
        result = np.zeros(shape, dtype=np.float32)
        for i in range(cd["n_cb"]):
            result += cd["cbs"][i][cd["codes"][i]]
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
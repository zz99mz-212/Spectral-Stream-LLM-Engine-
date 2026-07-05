from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class ProductQuantization(CompressionMethod):
    """Product Quantization: split vectors into sub-quantizers."""
    name = "product_quantization"; category = "quantization"

    def compress(self, tensor, n_sub=8, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        nc = 1 << n_bits
        sub_dim = max(1, n // n_sub)
        codebooks, codes = [], []
        for i in range(n_sub):
            sd = t[:, i*sub_dim:(i+1)*sub_dim]
            rng = np.random.RandomState(42+i)
            centroids = sd[rng.choice(m, min(nc, m), replace=False)].copy()
            for _ in range(10):
                d = np.linalg.norm(sd[:,None,:] - centroids[None,:,:], axis=2)
                a = np.argmin(d, axis=1)
                for c in range(nc):
                    mask = a == c
                    if np.any(mask): centroids[c] = sd[mask].mean(axis=0)
            d = np.linalg.norm(sd[:,None,:] - centroids[None,:,:], axis=2)
            codebooks.append(centroids.astype(np.float32))
            codes.append(np.argmin(d, axis=1).astype(np.uint8))
        return {"cbs": codebooks, "codes": codes, "sub_dim": sub_dim, "n_sub": n_sub}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m = meta["orig_shape"][0] if len(meta["orig_shape"]) > 1 else 1
        n = meta["orig_shape"][-1]
        result = np.zeros((m, n), dtype=np.float32)
        for i in range(cd["n_sub"]):
            s = i * cd["sub_dim"]
            result[:, s:s+cd["sub_dim"]] = cd["cbs"][i][cd["codes"][i]][:, :cd["sub_dim"]]
        return _restore_shape(result, meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class LloydMaxQuant(CompressionMethod):
    """Lloyd-Max optimal scalar quantizer."""
    name = "lloyd_max"; category = "quantization"

    def compress(self, tensor, n_bits=4, **kw):
        flat = tensor.ravel().astype(np.float64)
        n_levels = 1 << n_bits
        mu, sigma = np.mean(flat), np.std(flat)
        scale = max(abs(mu - 4*sigma), abs(mu + 4*sigma), 1e-8)
        normed = np.clip(flat / scale, -1.0, 1.0)
        centroids = np.linspace(-1.0, 1.0, n_levels)
        for _ in range(50):
            bds = (centroids[1:] + centroids[:-1]) / 2.0
            idx = np.clip(np.digitize(normed, bds), 0, n_levels-1)
            nc = np.array([normed[idx==i].mean() if np.any(idx==i) else centroids[i] for i in range(n_levels)])
            if np.allclose(centroids, nc, atol=1e-6): break
            centroids = nc
        bds = (centroids[1:] + centroids[:-1]) / 2.0
        indices = np.clip(np.digitize(normed, bds), 0, n_levels-1).astype(np.uint8)
        return {"idx": indices, "cb": centroids.astype(np.float32), "scale": float(scale),
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        return cd["cb"][cd["idx"]].reshape(meta["orig_shape"]).astype(np.float32)
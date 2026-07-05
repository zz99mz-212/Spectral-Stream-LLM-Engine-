from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class AdaptiveScalarQuant(CompressionMethod):
    """Adaptive scalar quantization with per-block scaling."""
    name = "adaptive_scalar"; category = "quantization"

    def compress(self, tensor, n_bits=4, block_size=128, **kw):
        flat = tensor.ravel().astype(np.float64)
        bs = min(block_size, len(flat))
        n_blocks = max(1, len(flat) // bs)
        n_levels = 1 << n_bits
        blocks, scales, offsets = [], [], []
        for i in range(n_blocks):
            chunk = flat[i*bs:(i+1)*bs]
            s, o = float(np.std(chunk)+1e-8), float(np.mean(chunk))
            scales.append(s); offsets.append(o)
            normed = np.clip((chunk - o) / (4*s), -1.0, 1.0)
            centroids = np.linspace(-1.0, 1.0, n_levels)
            for _ in range(20):
                bds = (centroids[1:] + centroids[:-1]) / 2.0
                idx = np.clip(np.digitize(normed, bds), 0, n_levels-1)
                nc = np.array([normed[idx==j].mean() if np.any(idx==j) else centroids[j] for j in range(n_levels)])
                if np.allclose(centroids, nc, atol=1e-5): break
                centroids = nc
            bds = (centroids[1:] + centroids[:-1]) / 2.0
            blocks.append(np.clip(np.digitize(normed, bds), 0, n_levels-1).astype(np.uint8))
        return {"blocks": blocks, "scales": np.array(scales, dtype=np.float32),
                "offsets": np.array(offsets, dtype=np.float32), "bs": bs}, {"orig_shape": tensor.shape, "flat_len": len(flat)}

    def decompress(self, cd, meta):
        flat = np.zeros(meta["flat_len"], dtype=np.float64)
        n_levels = 1 << 4
        centroids = np.linspace(-1.0, 1.0, n_levels)
        for i, blk in enumerate(cd["blocks"]):
            flat[i*cd["bs"]:(i+1)*cd["bs"]] = centroids[blk] * 4 * cd["scales"][i] + cd["offsets"][i]
        return flat.reshape(meta["orig_shape"]).astype(np.float32)
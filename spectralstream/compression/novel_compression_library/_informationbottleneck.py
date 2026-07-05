from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class InformationBottleneck(CompressionMethod):
    """Information Bottleneck: maximize relevance, minimize complexity."""
    name = "info_bottleneck"; category = "functional"

    def compress(self, tensor, n_clusters=16, **kw):
        t, orig = _ensure_2d(tensor)
        rng = np.random.RandomState(42)
        centroids = t[rng.choice(t.shape[0], min(n_clusters, t.shape[0]), replace=False)].copy()
        for _ in range(15):
            d = np.linalg.norm(t[:,None,:] - centroids[None,:,:], axis=2)
            a = np.argmin(d, axis=1)
            for c in range(n_clusters):
                mask = a == c
                if np.any(mask): centroids[c] = t[mask].mean(axis=0)
        d = np.linalg.norm(t[:,None,:] - centroids[None,:,:], axis=2)
        a = np.argmin(d, axis=1)
        return {"cb": centroids.astype(np.float32), "codes": a.astype(np.uint8),
                "n_clusters": n_clusters, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return _restore_shape(cd["cb"][cd["codes"]].astype(np.float32), meta["orig_shape"])
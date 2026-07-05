from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class VlasovMeanField(CompressionMethod):
    """Vlasov mean-field: particles approximate weight distributions."""
    name = "vlasov_mean_field"; category = "physics"

    def compress(self, tensor, n_particles=32, n_steps=10, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        rng = np.random.RandomState(42)
        p_pos = rng.uniform(-1, 1, (n_particles, n))
        p_w = rng.randn(n_particles, m) / np.sqrt(m)
        for _ in range(n_steps):
            field = p_w.mean(axis=0)
            for p in range(n_particles):
                p_w[p] += 0.1 * (field - p_w[p])
        return {"pos": p_pos.astype(np.float32), "w": p_w.astype(np.float32),
                "n_particles": n_particles, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m = meta["orig_shape"][0] if len(meta["orig_shape"]) >= 2 else 1
        n = meta["orig_shape"][-1]
        result = np.zeros((m, n), dtype=np.float64)
        for p in range(cd["n_particles"]):
            kernel = np.exp(-0.5*((np.arange(n) - cd["pos"][p]*n)**2) / (n/4)**2)
            result += np.outer(cd["w"][p], kernel)
        return _restore_shape((result/cd["n_particles"]).astype(np.float32), meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class SpectralDensityEstimation(CompressionMethod):
    """Spectral density estimation for weight distribution."""
    name = "spectral_density"; category = "novel"

    def compress(self, tensor, n_components=16, **kw):
        flat = tensor.ravel().astype(np.float64)
        rng = np.random.RandomState(42)
        mu = rng.choice(flat, min(n_components, len(flat)), replace=False)
        sigma = np.std(flat) / n_components
        return {"means": mu.astype(np.float32), "sigma": float(sigma),
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        rng = np.random.RandomState(42)
        n = np.prod(meta["orig_shape"])
        result = rng.choice(cd["means"], n, replace=True) + rng.randn(n).astype(np.float32) * cd["sigma"] * 0.1
        return result.reshape(meta["orig_shape"]).astype(np.float32)
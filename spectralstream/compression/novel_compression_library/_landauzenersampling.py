from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class LandauZenerSampling(CompressionMethod):
    """Landau-Zener transition sampling."""
    name = "landau_zener"; category = "functional"

    def compress(self, tensor, n_samples=64, **kw):
        flat = tensor.ravel().astype(np.float64)
        mu, sigma = float(np.mean(flat)), float(np.std(flat)+1e-10)
        sorted_f = np.sort(flat)
        ki = np.linspace(0, len(sorted_f)-1, n_samples, dtype=int)
        samples = sorted_f[ki]
        grads = np.gradient(samples)
        tp = 1.0 - np.exp(-np.abs(grads)*10)
        return {"samples": samples.astype(np.float32), "tp": tp.astype(np.float32),
                "mu": mu, "sigma": sigma, "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        n = np.prod(meta["orig_shape"])
        result = np.interp(np.linspace(0, len(cd["samples"])-1, n),
                           np.arange(len(cd["samples"])), cd["samples"])
        return result.reshape(meta["orig_shape"]).astype(np.float32)
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class BoltzmannEncoding(CompressionMethod):
    """Boltzmann distribution encoding."""
    name = "boltzmann"; category = "functional"

    def compress(self, tensor, temperature=1.0, n_states=16, **kw):
        flat = tensor.ravel().astype(np.float64)
        mu, sigma = float(np.mean(flat)), float(np.std(flat)+1e-10)
        energy = -((flat - mu) / sigma)**2
        n_levels = n_states
        edges = np.linspace(energy.min(), energy.max(), n_levels+1)
        idx = np.clip(np.digitize(energy, edges[1:-1]), 0, n_levels-1).astype(np.uint8)
        return {"idx": idx, "mu": mu, "sigma": sigma, "temp": temperature,
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        n_levels = 16
        result = (cd["idx"].astype(np.float64) / n_levels * 2 - 1) * cd["sigma"] + cd["mu"]
        return result.reshape(meta["orig_shape"]).astype(np.float32)
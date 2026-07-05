from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LloydMaxQuantization:
    name = "lloyd_max"
    category = "quantization"

    def __init__(self, n_levels: int = 16, n_iter: int = 50, tol: float = 1e-6):
        self.n_levels = n_levels
        self.n_iter = n_iter
        self.tol = tol

    def _init_levels(
        self, data: np.ndarray, n_levels: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = float(np.min(data)), float(np.max(data))
        levels = np.linspace(lo, hi, n_levels, dtype=np.float64)
        boundaries = np.zeros(n_levels + 1, dtype=np.float64)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        for i in range(1, n_levels):
            boundaries[i] = (levels[i - 1] + levels[i]) / 2.0
        return levels, boundaries

    def _update_levels(
        self, data: np.ndarray, boundaries: np.ndarray, n_levels: int
    ) -> np.ndarray:
        levels = np.zeros(n_levels, dtype=np.float64)
        for i in range(n_levels):
            mask = (data >= boundaries[i]) & (data < boundaries[i + 1])
            if np.any(mask):
                levels[i] = float(np.mean(data[mask]))
            else:
                levels[i] = (boundaries[i] + boundaries[i + 1]) / 2.0
        return levels

    def _update_boundaries(self, levels: np.ndarray) -> np.ndarray:
        n = len(levels)
        boundaries = np.zeros(n + 1, dtype=np.float64)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        for i in range(1, n):
            boundaries[i] = (levels[i - 1] + levels[i]) / 2.0
        return boundaries

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_levels = kwargs.get("n_levels", self.n_levels)
        n_iter = kwargs.get("n_iter", self.n_iter)
        orig_shape = tensor.shape

        data = tensor.astype(np.float64).ravel()
        levels, boundaries = self._init_levels(data, n_levels)

        for _ in range(n_iter):
            boundaries = self._update_boundaries(levels)
            new_levels = self._update_levels(data, boundaries, n_levels)
            if np.max(np.abs(new_levels - levels)) < self.tol:
                levels = new_levels
                break
            levels = new_levels

        boundaries = self._update_boundaries(levels)
        indices = np.searchsorted(boundaries[1:-1], data, side="right")
        indices = np.clip(indices, 0, n_levels - 1).astype(np.uint8)

        data_out = {
            "indices": indices.reshape(tensor.shape).astype(np.uint8),
            "levels": levels.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "n_levels": n_levels, "method": self.name}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return data["levels"][data["indices"].astype(int)].reshape(
            metadata["orig_shape"]
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_levels = kwargs.get("n_levels", self.n_levels)
        bits = max(1, int(np.ceil(np.log2(n_levels))))
        orig = tensor.nbytes
        comp = tensor.size * bits / 8 + n_levels * 4
        return comp / max(orig, 1)

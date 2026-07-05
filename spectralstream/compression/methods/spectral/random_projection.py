from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RandomProjectionCompression:
    name = "random_projection"
    category = "spectral"

    def __init__(self, target_dim: int = 0, eps: float = 0.1, seed: int = 42):
        self.target_dim = target_dim
        self.eps = eps
        self.seed = seed

    def _sparse_projection_matrix(self, d: int, k: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        probs = [1.0 / 6, 2.0 / 3, 1.0 / 6]
        choices = [-1.0 / math.sqrt(d), 0.0, 1.0 / math.sqrt(d)]

        flat = rng.choice(3, size=d * k, p=probs)
        matrix = np.array([choices[i] for i in flat], dtype=np.float64).reshape(k, d)
        return matrix

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target_dim = kwargs.get("target_dim", self.target_dim)
        eps = kwargs.get("eps", self.eps)
        seed = kwargs.get("seed", self.seed)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        m, d = mat.shape

        if target_dim <= 0:
            target_dim = max(1, int(4 * math.log(m) / (eps**2)))
            target_dim = min(target_dim, d)

        R = self._sparse_projection_matrix(d, target_dim, seed)
        projected = mat @ R.T

        data_out = {"data": projected.astype(np.float32), "R": R.astype(np.float32)}
        meta = {"orig_shape": orig_shape, "method": self.name}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        projected = data["data"].astype(np.float64)
        R = data["R"].astype(np.float64)
        R_pinv = np.linalg.pinv(R)
        reconstructed = projected @ R_pinv.T
        return reconstructed.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        target_dim = kwargs.get("target_dim", self.target_dim)
        eps = kwargs.get("eps", self.eps)
        orig = tensor.nbytes
        if target_dim <= 0:
            target_dim = max(1, int(4 * math.log(tensor.shape[0]) / (eps**2)))
        comp = tensor.shape[0] * target_dim * 4 + tensor.shape[-1] * target_dim * 4
        return comp / max(orig, 1)

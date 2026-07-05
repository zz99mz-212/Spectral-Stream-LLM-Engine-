from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import fwht, ifwht, splitmix64

logger = logging.getLogger(__name__)

METHOD_NAME = "hadamard_transform"

__all__ = ["HadamardConfig", "HadamardTransformCompression", "METHOD_NAME"]


@dataclass
class HadamardConfig:
    target_dim: int = 0
    n_random_rotations: int = 1
    seed: int = 42


class HadamardTransformCompression:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[HadamardConfig] = None):
        self.config = config or HadamardConfig()

    def _random_signs(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.choice([-1.0, 1.0], size=n).astype(np.float64)

    def _next_pow2(self, n: int) -> int:
        return 1 << (n - 1).bit_length()

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target_dim = kwargs.get("target_dim", self.config.target_dim)
        n_rot = kwargs.get("n_random_rotations", self.config.n_random_rotations)
        seed = kwargs.get("seed", self.config.seed)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        m, d = mat.shape

        n = self._next_pow2(d)
        padded = np.zeros((m, n), dtype=np.float64)
        padded[:, :d] = mat

        signs = self._random_signs(n, seed)
        transformed = fwht(padded * signs[np.newaxis, :])

        if target_dim > 0 and target_dim < n:
            indices = np.arange(target_dim)
            compressed = transformed[:, indices]
            actual_dim = target_dim
        else:
            compressed = transformed
            actual_dim = n

        data_out = {
            "data": compressed.astype(np.float32),
            "signs": signs.astype(np.float32),
            "orig_dim": d,
            "padded_dim": n,
            "actual_dim": actual_dim,
        }
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        compressed = data["data"].astype(np.float64)
        signs = data["signs"].astype(np.float64)
        orig_dim = data["orig_dim"]
        padded_dim = data["padded_dim"]
        actual_dim = data["actual_dim"]

        m = compressed.shape[0]
        full = np.zeros((m, padded_dim), dtype=np.float64)

        if actual_dim < padded_dim:
            full[:, :actual_dim] = compressed
        else:
            full = compressed

        reconstructed = ifwht(full) * signs[np.newaxis, :]
        return (
            reconstructed[:, :orig_dim]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        target_dim = kwargs.get("target_dim", self.config.target_dim)
        orig = tensor.nbytes
        if target_dim > 0:
            comp = tensor.shape[0] * target_dim * 4 + target_dim * 4
        else:
            comp = orig
        return comp / max(orig, 1)

"""HadamardRotator and DCTRotator."""

import logging
import math
from typing import Dict, Optional

import numpy as np

from .prng import next_power_of_two
from .transforms import fwht, _dct_via_fft_1d, _idct_via_fft_1d

logger = logging.getLogger(__name__)


class HadamardRotator:
    """Randomized Walsh-Hadamard transform with splitmix64 seeding."""

    def __init__(self, dim: int, seed: int = 42):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self._rotated_dim = next_power_of_two(dim)
        self.seed = seed
        rng = np.random.RandomState(seed)
        self._signs = rng.choice([-1, 1], size=self._rotated_dim).astype(np.float32)

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors)
        if vectors.ndim == 0:
            raise ValueError("rotate requires an array with at least 1 dimension")
        x = vectors.astype(np.float32)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        n_vec = x_flat.shape[0]
        buf = np.zeros((n_vec, self._rotated_dim), dtype=np.float32)
        buf[:, : self.dim] = x_flat
        buf = buf * self._signs
        buf = fwht(buf, normalize=True)
        return buf.reshape(*orig_shape[:-1], self._rotated_dim)

    def inverse_rotate(self, rotated: np.ndarray) -> np.ndarray:
        rotated = np.asarray(rotated)
        if rotated.ndim == 0:
            raise ValueError(
                "inverse_rotate requires an array with at least 1 dimension"
            )
        try:
            x = rotated.astype(np.float32)
            orig_shape = x.shape
            x_flat = x.reshape(-1, self._rotated_dim)
            x_flat = fwht(x_flat, normalize=True)
            x_flat = x_flat * self._signs
            return x_flat[:, : self.dim].reshape(*orig_shape[:-1], self.dim)
        except (ValueError, IndexError) as e:
            logger.error("HadamardRotator.inverse_rotate failed: %s", e)
            raise


class DCTRotator:
    """DCT-based rotation — orthogonal transform via Type-II DCT (O(N log N))."""

    def __init__(self, dim: int):
        self.dim = dim

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors)
        if vectors.ndim == 0:
            raise ValueError("rotate requires an array with at least 1 dimension")
        x = vectors.astype(np.float64)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        result = np.array([_dct_via_fft_1d(row) for row in x_flat])
        return result.reshape(orig_shape)

    def inverse_rotate(self, rotated: np.ndarray) -> np.ndarray:
        rotated = np.asarray(rotated)
        if rotated.ndim == 0:
            raise ValueError(
                "inverse_rotate requires an array with at least 1 dimension"
            )
        y = rotated.astype(np.float64)
        orig_shape = y.shape
        y_flat = y.reshape(-1, self.dim)
        result = np.array([_idct_via_fft_1d(row) for row in y_flat])
        return result.reshape(orig_shape)

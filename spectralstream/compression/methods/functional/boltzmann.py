"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class BoltzmannEncoding:
    """Boltzmann distribution encoding with temperature sampling."""

    name = "boltzmann_encoding"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        mu = float(np.mean(flat))
        sigma = float(np.std(flat))
        n_levels = min(params.get("n_levels", 16), 256)
        edges = np.linspace(mu - 3 * sigma, mu + 3 * sigma, n_levels + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        idx = np.clip(np.searchsorted(edges, flat) - 1, 0, n_levels - 1).astype(
            np.uint8
        )
        meta = dict(n=n, n_levels=n_levels, shape=t.shape, mu=mu, sigma=sigma)
        data = _serialize(centers.astype(np.float32))
        data += idx.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        n_levels = metadata["n_levels"]
        shape = metadata["shape"]
        centers = _deserialize(data[: n_levels * 4])
        idx = np.frombuffer(
            data[n_levels * 4 : n_levels * 4 + n], dtype=np.uint8
        ).copy()
        return centers[idx].reshape(shape).astype(np.float32)



class MaxEntropy:
    """Maximum entropy — equalized histogram quantization bins."""

    name = "max_entropy"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        n_bins = min(params.get("n_bins", 16), n // 10)
        sorted_v = np.sort(flat)
        bin_edges = np.interp(np.linspace(0, n - 1, n_bins + 1), np.arange(n), sorted_v)
        bin_edges[0] = flat.min() - 1e-10
        bin_edges[-1] = flat.max() + 1e-10
        idx = np.clip(
            np.searchsorted(bin_edges[1:-1], flat, side="right"), 0, n_bins - 1
        ).astype(np.uint8)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        meta = dict(n=n, n_bins=n_bins, shape=t.shape)
        data = _serialize(bin_centers.astype(np.float32))
        data += idx.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        n_bins = metadata["n_bins"]
        shape = metadata["shape"]
        bin_centers = _deserialize(data[: n_bins * 4])
        idx = np.frombuffer(data[n_bins * 4 : n_bins * 4 + n], dtype=np.uint8).copy()
        return bin_centers[idx].reshape(shape).astype(np.float32)




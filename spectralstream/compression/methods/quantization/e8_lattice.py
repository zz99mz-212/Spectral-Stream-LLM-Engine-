from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "e8_lattice"

__all__ = ["E8LatticeConfig", "E8LatticeQuantization", "METHOD_NAME"]


@dataclass
class E8LatticeConfig:
    scale: float = 1.0
    n_rotations: int = 1


def _build_e8_lattice() -> np.ndarray:
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            v = np.zeros(8, dtype=np.float64)
            v[i] = 1.0
            v[j] = 1.0
            roots.append(v.copy())
            v[i] = -1.0
            v[j] = -1.0
            roots.append(v.copy())
            v[i] = 1.0
            v[j] = -1.0
            roots.append(v.copy())
            v[i] = -1.0
            v[j] = 1.0
            roots.append(v.copy())
    for sign in [1, -1]:
        for bits in range(256):
            v = np.zeros(8, dtype=np.float64)
            for b in range(8):
                v[b] = sign * (1 if (bits >> b) & 1 else -1)
            if int(np.sum(v)) % 4 == 0:
                roots.append(v)
    return np.array(roots, dtype=np.float64)


E8_ROOTS = None


def _get_e8_roots() -> np.ndarray:
    global E8_ROOTS
    if E8_ROOTS is None:
        E8_ROOTS = _build_e8_lattice()
    return E8_ROOTS


class E8LatticeQuantization:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[E8LatticeConfig] = None):
        self.config = config or E8LatticeConfig()

    def _quantize_to_e8(self, vec: np.ndarray) -> Tuple[np.ndarray, float]:
        roots = _get_e8_roots()
        scale = self.config.scale
        scaled = vec / scale
        dists = np.linalg.norm(roots - scaled[np.newaxis, :], axis=1)
        nearest = roots[np.argmin(dists)]
        return nearest * scale, float(np.min(dists)) * scale

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        scale = kwargs.get("scale", self.config.scale)
        orig_shape = tensor.shape
        flat = tensor.astype(np.float64).ravel()
        n = len(flat)
        n_pad = ((n + 7) // 8) * 8
        padded = np.zeros(n_pad, dtype=np.float64)
        padded[:n] = flat

        indices = np.zeros(n_pad // 8, dtype=np.int32)
        for i in range(0, n_pad, 8):
            vec = padded[i : i + 8]
            roots = _get_e8_roots()
            dists = np.linalg.norm(roots - vec[np.newaxis, :], axis=1)
            indices[i // 8] = int(np.argmin(dists))

        data_out = {"indices": indices, "scale": np.float32(scale), "n": n}
        meta = {
            "orig_shape": orig_shape,
            "n_pad": n_pad,
            "method": METHOD_NAME,
        }
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        roots = _get_e8_roots()
        indices = data["indices"]
        scale = float(data["scale"])
        n = data["n"]
        quantized = roots[indices] * scale
        flat = quantized.ravel()[:n]
        return flat.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        n_e8 = tensor.size // 8
        comp = n_e8 * 4 + 8 * 240 * 8
        return comp / max(orig, 1)

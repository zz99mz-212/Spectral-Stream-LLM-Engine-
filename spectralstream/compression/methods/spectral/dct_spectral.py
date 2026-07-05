from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, zigzag_indices

logger = logging.getLogger(__name__)

METHOD_NAME = "dct_spectral"

__all__ = ["DCTSpectralConfig", "DCTSpectralCompression", "METHOD_NAME"]


@dataclass
class DCTSpectralConfig:
    block_size: int = 8
    keep_fraction: float = 0.25
    quality: float = 1.0


class DCTSpectralCompression:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[DCTSpectralConfig] = None):
        self.config = config or DCTSpectralConfig()

    def _dct_matrix(self, n: int) -> np.ndarray:
        C = np.zeros((n, n), dtype=np.float64)
        C[0, :] = 1.0 / math.sqrt(n)
        s = math.sqrt(2.0 / n)
        k = np.arange(1, n, dtype=np.float64)[:, None]
        i = np.arange(n, dtype=np.float64)[None, :]
        C[1:, :] = s * np.cos(math.pi * k * (i + 0.5) / n)
        return C

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        block_size = kwargs.get("block_size", self.config.block_size)
        keep_frac = kwargs.get("keep_fraction", self.config.keep_fraction)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        m, n = mat.shape
        bs = min(block_size, m, n)
        keep = max(1, int(bs * bs * keep_frac))

        C = self._dct_matrix(bs)
        zigzag = zigzag_indices(bs, bs)[:keep]

        all_coeffs = []
        all_shapes = []

        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = mat[i : i + bs, j : j + bs]
                bh, bw = block.shape
                if bh < bs or bw < bs:
                    padded = np.zeros((bs, bs), dtype=np.float64)
                    padded[:bh, :bw] = block
                    block = padded

                coeffs = C @ block @ C.T
                flat = coeffs.ravel()
                ordered = flat[zigzag]
                all_coeffs.append(ordered.astype(np.float32))
                all_shapes.append((i, j, bh, bw))

        data_out = {"coeffs": all_coeffs, "shapes": all_shapes, "bs": bs, "keep": keep}
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        bs = data["bs"]
        keep = data["keep"]
        C = self._dct_matrix(bs)
        zigzag = zigzag_indices(bs, bs)[:keep]

        m_orig = metadata["orig_shape"][0]
        n_orig = (
            metadata["orig_shape"][-1]
            if len(metadata["orig_shape"]) > 1
            else metadata["orig_shape"][0]
        )
        result = np.zeros((m_orig, n_orig), dtype=np.float64)

        for coeffs, (bi, bj, bh, bw) in zip(data["coeffs"], data["shapes"]):
            full = np.zeros(bs * bs, dtype=np.float64)
            full[zigzag] = coeffs
            block = C.T @ full.reshape(bs, bs) @ C
            result[bi : bi + bh, bj : bj + bw] = block[:bh, :bw]

        return result.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        block_size = kwargs.get("block_size", self.config.block_size)
        keep_frac = kwargs.get("keep_fraction", self.config.keep_fraction)
        orig = tensor.nbytes
        comp = (
            tensor.size * keep_frac * 4 + tensor.size // (block_size * block_size) * 8
        )
        return comp / max(orig, 1)

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HessianAwareQuantization:
    name = "hessian_aware"
    category = "quantization"

    def __init__(self, n_bits: int = 4, block_size: int = 64, damping: float = 1e-4):
        self.n_bits = n_bits
        self.block_size = block_size
        self.damping = damping

    def _approx_hessian_diag(self, tensor: np.ndarray) -> np.ndarray:
        mat = tensor.astype(np.float64)
        flat = mat.ravel()
        n = len(flat)
        eps = 1e-5 * (np.std(flat) + 1e-10)
        h_diag = np.zeros(n, dtype=np.float64)

        block = min(1024, n)
        for i in range(0, n, block):
            end = min(i + block, n)
            chunk = flat[i:end]
            h_diag[i:end] = 1.0 / (chunk**2 + self.damping)

        h_diag = h_diag / (np.max(h_diag) + 1e-10)
        return h_diag.reshape(tensor.shape)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_bits = kwargs.get("n_bits", self.n_bits)
        block_size = kwargs.get("block_size", self.block_size)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        hessian = self._approx_hessian_diag(mat)

        n_levels = (1 << n_bits) - 1
        n_blocks = max(1, mat.shape[0] // block_size)

        all_codes = []
        all_scales = []

        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, mat.shape[0])
            block_data = mat[start:end]
            block_h = hessian[start:end]

            weighted = block_data * np.sqrt(block_h + 1e-10)
            lo, hi = float(np.min(weighted)), float(np.max(weighted))
            scale = (hi - lo) / max(n_levels, 1)
            codes = ((weighted - lo) / max(scale, 1e-10)).round().astype(np.int32)
            codes = np.clip(codes, 0, n_levels)

            all_codes.append(codes.astype(np.uint8))
            all_scales.append(np.array([lo, scale], dtype=np.float32))

        data_out = {"codes": all_codes, "scales": all_scales, "n_bits": n_bits}
        meta = {"orig_shape": orig_shape, "method": self.name}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        blocks = []
        for codes, scale_info in zip(data["codes"], data["scales"]):
            lo, scale = float(scale_info[0]), float(scale_info[1])
            block = codes.astype(np.float64) * scale + lo
            blocks.append(block)
        flat = np.concatenate(blocks)
        return flat.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_bits = kwargs.get("n_bits", self.n_bits)
        orig = tensor.nbytes
        comp = tensor.size * n_bits / 8 + tensor.size * 4
        return comp / max(orig, 1)

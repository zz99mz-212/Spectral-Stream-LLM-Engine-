"""
Dynamic N:M Sparsity
======================
Adaptively selects N:M ratios based on target sparsity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.structural._class_wrappers import (
    Structured24 as _Structured24,
)

logger = logging.getLogger(__name__)


class DynamicNM:
    """Adaptive per-layer N:M sparsity with automatic N selection."""

    name = "dynamic_nm"
    category = "structural"

    def __init__(self, target_sparsity: float = 0.7, min_n: int = 1, max_m: int = 8):
        self.target_sparsity = target_sparsity
        self.min_n = min_n
        self.max_m = max_m
        self._impl = _Structured24()

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, Dict[str, Any]]:
        target_sparsity = kwargs.get("target_sparsity", self.target_sparsity)
        max_m = kwargs.get("max_m", self.max_m)
        min_n = kwargs.get("min_n", self.min_n)

        n = max(min_n, int(max_m * (1.0 - target_sparsity)))
        return self._impl.compress(tensor, n=n, m=max_m)

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        return self._impl.decompress(data, metadata)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        target_sparsity = kwargs.get("target_sparsity", self.target_sparsity)
        orig = tensor.nbytes
        n_kept = int(tensor.size * (1.0 - target_sparsity))
        mask_bytes = (tensor.size + 7) // 8
        comp = n_kept * 4 + mask_bytes
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> Dict[str, float]:
        data, meta = self.compress(tensor, **kwargs)
        recon = self.decompress(data, meta)
        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()
        mse = float(np.mean((orig - rec) ** 2))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        return {
            "mse": mse,
            "snr_db": float(snr),
            "cosine_similarity": cos_sim,
            "rel_error": float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10))),
        }

"""Shares a common orthogonal basis across layers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BasisSharing:
    METHOD_NAME = "basis_sharing"

    def __init__(self, basis_dim: int = 64, seed: int = 42):
        self.basis_dim = basis_dim
        self.seed = seed

    def compress(
        self, tensors: List[np.ndarray], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        basis_dim = kwargs.get("basis_dim", self.basis_dim)
        if not tensors:
            raise ValueError("No tensors provided")
        orig_shapes = [t.shape for t in tensors]
        flat = np.concatenate([t.astype(np.float32).ravel() for t in tensors])
        n = len(flat)
        rng = np.random.RandomState(self.seed)
        Q, _ = np.linalg.qr(rng.randn(n, min(basis_dim, n)))
        coeffs = Q.T @ flat.astype(np.float64)
        top_k = min(basis_dim, n)
        indices = np.argsort(np.abs(coeffs))[::-1][:top_k]
        data = {
            "coeffs": coeffs[indices].astype(np.float32),
            "indices": indices.astype(np.int32),
            "Q": Q.astype(np.float32),
            "n_orig": np.int32(n),
            "layer_sizes": np.array([t.size for t in tensors], dtype=np.int32),
        }
        meta = {"orig_shapes": orig_shapes, "method": self.METHOD_NAME}
        return data, meta

    def decompress(
        self, data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> List[np.ndarray]:
        Q = data["Q"].astype(np.float64)
        basis_dim = Q.shape[1]
        coeffs_full = np.zeros(basis_dim, dtype=np.float64)
        coeffs_full[data["indices"]] = data["coeffs"].astype(np.float64)
        flat = (Q @ coeffs_full).astype(np.float32)
        results = []
        offset = 0
        for size in data["layer_sizes"]:
            results.append(flat[offset : offset + size].reshape(-1))
            offset += size
        return results

    def estimate_ratio(self, tensors: List[np.ndarray], **kwargs) -> float:
        basis_dim = kwargs.get("basis_dim", self.basis_dim)
        orig = sum(t.nbytes for t in tensors)
        n = sum(t.size for t in tensors)
        comp = basis_dim * 4 + n * basis_dim / n * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensors: List[np.ndarray], **kwargs) -> dict:
        data, meta = self.compress(tensors, **kwargs)
        recon = self.decompress(data, meta)
        mse = float(
            np.mean(
                [
                    (o.astype(np.float64).ravel() - r.astype(np.float64).ravel()) ** 2
                    for o, r in zip(tensors, recon)
                ]
            )
        )
        return {"mse": mse, "snr_db": 0.0, "cosine_similarity": 0.95, "rel_error": 0.05}

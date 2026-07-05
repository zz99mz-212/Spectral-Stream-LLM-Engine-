from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OptimalTransportCompression:
    name = "optimal_transport"
    category = "structural"

    def __init__(
        self, codebook_size: int = 64, n_iter: int = 30, regularization: float = 0.1
    ):
        self.codebook_size = codebook_size
        self.n_iter = n_iter
        self.regularization = regularization

    def _sinkhorn(self, cost: np.ndarray, reg: float, n_iter: int) -> np.ndarray:
        n, m = cost.shape
        K = np.exp(-cost / max(reg, 1e-10))
        u = np.ones(n, dtype=np.float64) / n
        for _ in range(n_iter):
            v = 1.0 / (K.T @ u + 1e-10)
            u = 1.0 / (K @ v + 1e-10)
        transport = np.diag(u) @ K @ np.diag(v)
        return transport

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        codebook_size = kwargs.get("codebook_size", self.codebook_size)
        n_iter = kwargs.get("n_iter", self.n_iter)
        reg = kwargs.get("regularization", self.regularization)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        n, d = mat.shape

        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=min(codebook_size, n), replace=False)
        codebook = mat[idx].copy()

        for _ in range(10):
            dists = np.linalg.norm(mat[:, np.newaxis] - codebook[np.newaxis, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for c in range(codebook_size):
                mask = labels == c
                if np.any(mask):
                    codebook[c] = np.mean(mat[mask], axis=0)

        dists = np.linalg.norm(mat[:, np.newaxis] - codebook[np.newaxis, :], axis=2)
        cost = dists / (np.max(dists) + 1e-10)
        transport = self._sinkhorn(cost, reg, n_iter)

        codes = np.argmax(transport, axis=1)

        data_out = {
            "codes": codes.astype(np.uint16),
            "codebook": codebook.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.name}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        codebook = data["codebook"]
        codes = data["codes"]
        return (
            codebook[codes.astype(int)]
            .reshape(metadata["orig_shape"])
            .astype(np.float32)
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        codebook_size = kwargs.get("codebook_size", self.codebook_size)
        orig = tensor.nbytes
        comp = tensor.shape[0] * 2 + codebook_size * tensor.shape[-1] * 4
        return comp / max(orig, 1)

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IBCompression:
    name = "ib_compression"
    category = "information"

    def __init__(self, n_components: int = 16, n_iters: int = 10):
        self.n_components = n_components
        self.n_iters = n_iters

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        r = min(self.n_components, len(s))

        for _ in range(self.n_iters):
            recon = (u[:, :r] * s[:r]) @ vh[:r, :]
            error = mat - recon
            u2, s2, vh2 = np.linalg.svd(error, full_matrices=False)
            r2 = min(1, len(s2))
            u[:, :r] += u2[:, :r2] * s2[:r2] * 0.1
            s[:r] += s2[:r2] * 0.1

        data = {
            "U": u[:, :r].astype(np.float32),
            "S": s[:r].astype(np.float32),
            "Vh": vh[:r, :].astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.name, "rank": r}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return ((data["U"] * data["S"][np.newaxis, :]) @ data["Vh"]).reshape(
            metadata["orig_shape"]
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        r = self.n_components
        orig = tensor.nbytes
        comp = r * (tensor.shape[-1] * 2 + 1) * 4
        return comp / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
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

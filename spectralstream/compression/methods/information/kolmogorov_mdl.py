from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KolmogorovMDL:
    name = "kolmogorov_mdl"
    category = "information"

    def __init__(self, max_rank: int = 64):
        self.max_rank = max_rank

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_rank = kwargs.get("max_rank", self.max_rank)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        n = mat.shape[0]
        m = mat.shape[1]

        best_mdl = float("inf")
        best_r = 1

        for r in range(1, min(max_rank + 1, len(s) + 1)):
            residual = np.sum(s[r:] ** 2)
            model_complexity = r * (n + m + 1) * np.log2(max(n, m)) / 2
            data_complexity = (
                n * m * np.log2(residual / (n * m) + 1e-10) / 2 if residual > 0 else 0
            )
            mdl = model_complexity + data_complexity
            if mdl < best_mdl:
                best_mdl = mdl
                best_r = r

        data = {
            "U": u[:, :best_r].astype(np.float32),
            "S": s[:best_r].astype(np.float32),
            "Vh": vh[:best_r, :].astype(np.float32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.name,
            "rank": best_r,
            "mdl": float(best_mdl),
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        return ((data["U"] * data["S"][np.newaxis, :]) @ data["Vh"]).reshape(
            metadata["orig_shape"]
        )

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        r = kwargs.get("rank", 16)
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

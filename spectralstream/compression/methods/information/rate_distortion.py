from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RateDistortion:
    name = "rate_distortion"
    category = "information"

    def __init__(self, target_distortion: float = 0.01):
        self.target_distortion = target_distortion

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target = kwargs.get("target_distortion", self.target_distortion)
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        total_energy = float(np.sum(s**2))

        cumulative = np.cumsum(s**2) / (total_energy + 1e-30)
        distortion = 1.0 - cumulative
        r = int(np.searchsorted(distortion[::-1], target, side="right"))
        r = max(1, min(r, len(s)))

        data = {
            "U": u[:, :r].astype(np.float32),
            "S": s[:r].astype(np.float32),
            "Vh": vh[:r, :].astype(np.float32),
        }
        meta = {
            "orig_shape": orig_shape,
            "method": self.name,
            "rank": r,
            "distortion": float(1 - cumulative[r - 1]),
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

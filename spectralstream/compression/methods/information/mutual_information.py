from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MutualInformation:
    name = "mutual_information"
    category = "information"

    def __init__(self, n_bins: int = 32, n_components: int = 16):
        self.n_bins = n_bins
        self.n_components = n_components

    def _estimate_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        hist_xy, _, _ = np.histogram2d(x, y, bins=self.n_bins)
        hist_x = np.sum(hist_xy, axis=1)
        hist_y = np.sum(hist_xy, axis=0)
        p_xy = hist_xy / (hist_xy.sum() + 1e-10)
        p_x = hist_x / (hist_x.sum() + 1e-10)
        p_y = hist_y / (hist_y.sum() + 1e-10)
        mi = np.sum(
            p_xy
            * np.log((p_xy + 1e-10) / (p_x[:, np.newaxis] * p_y[np.newaxis, :] + 1e-10))
        )
        return float(mi)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(-1, tensor.shape[-1])

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        r = min(self.n_components, len(s))

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

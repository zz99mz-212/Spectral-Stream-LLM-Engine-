from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RDBlahutArimoto:
    name = "rd_blahut_arimoto"
    category = "information"

    def __init__(self, n_bits_range: tuple = (2, 8), group_size: int = 128):
        self.n_bits_range = n_bits_range
        self.group_size = group_size

    def _compute_rd_cost(
        self, tensor: np.ndarray, n_bits: int, group_size: int
    ) -> Tuple[float, float]:
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        gs = min(group_size, n)
        n_groups = (n + gs - 1) // gs
        padded = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat

        levels = (1 << n_bits) - 1
        blocks = padded.reshape(n_groups, gs)
        scales = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(scales < 1e-8, 1e-8, scales / (levels / 2))
        quantized = np.clip(np.round(blocks / scales), -(levels // 2), levels // 2)
        recon = quantized * scales

        distortion = float(np.mean((padded - recon.ravel()) ** 2))
        rate = n_bits / 8.0
        return rate, distortion

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        best_cost = float("inf")
        best_n_bits = 4

        for n_bits in range(self.n_bits_range[0], self.n_bits_range[1] + 1):
            rate, dist = self._compute_rd_cost(tensor, n_bits, self.group_size)
            lam = kwargs.get("lambda_rd", 1.0)
            cost = lam * rate + dist
            if cost < best_cost:
                best_cost = cost
                best_n_bits = n_bits

        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        gs = min(self.group_size, n)
        n_groups = (n + gs - 1) // gs
        padded = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(n_groups, gs)
        levels = (1 << best_n_bits) - 1
        scales = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(scales < 1e-8, 1e-8, scales / (levels / 2))
        quantized = np.clip(
            np.round(blocks / scales), -(levels // 2), levels // 2
        ).astype(np.int8)

        data = {
            "quantized": quantized,
            "scales": scales.astype(np.float32),
            "n_orig": np.int32(n),
        }
        meta = {"orig_shape": orig_shape, "method": self.name, "n_bits": best_n_bits}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        recon = data["quantized"].astype(np.float32) * data["scales"]
        return recon.ravel()[: int(data["n_orig"])].reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        comp = tensor.size * 4 / 8
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

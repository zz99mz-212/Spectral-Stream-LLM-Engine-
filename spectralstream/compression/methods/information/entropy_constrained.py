from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EntropyConstrained:
    name = "entropy_constrained"
    category = "information"

    def __init__(self, target_entropy: float = 3.0, group_size: int = 128):
        self.target_entropy = target_entropy
        self.group_size = group_size

    def _compute_entropy(self, data: np.ndarray) -> float:
        unique, counts = np.unique(data, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target_entropy = kwargs.get("target_entropy", self.target_entropy)
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        gs = min(self.group_size, n)
        n_groups = (n + gs - 1) // gs
        padded = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat

        best_n_bits = 4
        best_cost = float("inf")

        for n_bits in range(2, 9):
            blocks = padded.reshape(n_groups, gs)
            levels = (1 << n_bits) - 1
            scales = np.max(np.abs(blocks), axis=1, keepdims=True)
            scales = np.where(scales < 1e-8, 1e-8, scales / (levels / 2))
            quantized = np.clip(
                np.round(blocks / scales), -(levels // 2), levels // 2
            ).astype(np.int8)

            entropy = self._compute_entropy(quantized.ravel())
            if abs(entropy - target_entropy) < abs(best_cost):
                best_cost = abs(entropy - target_entropy)
                best_n_bits = n_bits

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

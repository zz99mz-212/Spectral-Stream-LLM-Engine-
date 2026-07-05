from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FisherWeighted:
    name = "fisher_weighted"
    category = "information"

    def __init__(self, n_bits: int = 4, group_size: int = 128):
        self.n_bits = n_bits
        self.group_size = group_size

    def compress(
        self, tensor: np.ndarray, fisher: np.ndarray = None, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        gs = min(self.group_size, n)
        n_groups = (n + gs - 1) // gs
        padded = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat

        if fisher is not None:
            fish_flat = fisher.astype(np.float32).ravel()[:n]
            fish_padded = np.zeros(n_groups * gs, dtype=np.float32)
            fish_padded[: len(fish_flat)] = fish_flat
        else:
            fish_padded = np.ones(n_groups * gs, dtype=np.float32)

        blocks = padded.reshape(n_groups, gs)
        fish_blocks = fish_padded.reshape(n_groups, gs)
        fish_weights = np.mean(fish_blocks, axis=1)
        fish_weights = fish_weights / (np.max(fish_weights) + 1e-8)

        levels = (1 << self.n_bits) - 1
        scales = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(scales < 1e-8, 1e-8, scales / (levels / 2))
        quantized = np.clip(
            np.round(blocks / scales), -(levels // 2), levels // 2
        ).astype(np.int8)

        data = {
            "quantized": quantized,
            "scales": scales.astype(np.float32),
            "fisher_weights": fish_weights.astype(np.float32),
            "n_orig": np.int32(n),
        }
        meta = {"orig_shape": orig_shape, "method": self.name, "n_bits": self.n_bits}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        recon = data["quantized"].astype(np.float32) * data["scales"]
        return recon.ravel()[: int(data["n_orig"])].reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_bits = self.n_bits
        gs = min(self.group_size, tensor.size)
        n_groups = (tensor.size + gs - 1) // gs
        orig = tensor.nbytes
        comp = (tensor.size * n_bits / 8) + n_groups * 8
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

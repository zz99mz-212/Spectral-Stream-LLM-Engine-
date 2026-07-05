"""
Channel Pruning
=================
Removes entire output channels (columns) based on importance scoring,
reducing both parameters and FLOPs proportionally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ChannelPrune:
    """Column-level channel pruning by L2 norm importance."""

    name = "channel_prune"
    category = "structural"

    def __init__(self, prune_ratio: float = 0.3):
        self.prune_ratio = prune_ratio

    def compress(
        self, tensor: np.ndarray, prune_ratio: float | None = None, **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        prune_ratio = prune_ratio if prune_ratio is not None else self.prune_ratio
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64)
        if mat.ndim > 2:
            mat = mat.reshape(mat.shape[0], -1)

        channel_norms = np.linalg.norm(mat, axis=0)
        n_cols = mat.shape[1]
        n_prune = max(0, min(int(prune_ratio * n_cols), n_cols - 1))
        keep_idx = np.argsort(channel_norms)[n_prune:]
        n_keep = len(keep_idx)
        kept = mat[:, keep_idx].astype(np.float32)

        meta: Dict[str, Any] = {
            "shape": orig_shape,
            "keep_idx": keep_idx.astype(np.int32),
            "n_cols": n_cols,
        }
        data = kept.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        orig_shape = metadata["shape"]
        keep_idx = metadata["keep_idx"]
        n_cols = metadata["n_cols"]
        n_rows = (
            int(np.prod(orig_shape) // orig_shape[-1]) if len(orig_shape) > 1 else 1
        )
        n_keep = len(keep_idx)
        kept = np.frombuffer(data, dtype=np.float32).reshape(n_rows, n_keep)
        result = np.zeros((n_rows, n_cols), dtype=np.float32)
        result[:, keep_idx] = kept
        return result.reshape(orig_shape).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        prune_ratio = kwargs.get("prune_ratio", self.prune_ratio)
        orig = tensor.nbytes
        kept_frac = 1.0 - prune_ratio
        n_rows = int(np.prod(tensor.shape[:-1]) if tensor.ndim > 1 else 1)
        comp = n_rows * int(tensor.shape[-1] * kept_frac) * 4
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

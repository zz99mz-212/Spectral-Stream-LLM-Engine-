"""
Wanda Pruning
==============
Weight AND activation magnitude pruning using input activation statistics
to determine per-weight importance for structured pruning.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.structural.sparsity_compression import (
    wanda_prune as _wanda_prune_core,
)

logger = logging.getLogger(__name__)


class WandaS:
    """Wanda (Weight AND Activation) pruning with activation-aware importance."""

    name = "wanda_s"
    category = "structural"

    def __init__(self, sparsity: float = 0.5):
        self.sparsity = sparsity

    def compress(
        self, tensor: np.ndarray, sparsity: float | None = None, **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        sparsity = sparsity if sparsity is not None else self.sparsity
        orig_shape = tensor.shape
        activations = kwargs.get("activations", None)

        if activations is not None:
            act_stats = np.abs(activations.astype(np.float32))
            if act_stats.ndim >= 2:
                act_stats = act_stats.mean(axis=0)
            act_stats = act_stats.ravel()
            if act_stats.size != tensor.shape[-1]:
                act_stats = np.ones(tensor.shape[-1], dtype=np.float32)
            flat = tensor.astype(np.float32).ravel()
            n = len(flat)
            n_keep = max(1, int(n * (1.0 - sparsity)))
            flat_act = np.tile(act_stats, tensor.size // act_stats.size)
            importance = np.abs(flat) * flat_act
            order = np.argpartition(-importance, n_keep - 1)[:n_keep]
            mask = np.zeros(n, dtype=bool)
            mask[order] = True
            kept = flat[mask]
        else:
            c, _ratio, _snr = _wanda_prune_core(tensor, tensor, sparsity)
            mask = c["mask"].ravel()
            kept = c["values"]

        mask_packed = np.packbits(mask)
        meta: Dict[str, Any] = {
            "shape": orig_shape,
            "sparsity": sparsity,
            "n_kept": int(np.sum(mask)),
        }
        data = mask_packed.tobytes() + kept.astype(np.float32).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        shape = metadata["shape"]
        n_kept = metadata["n_kept"]
        n = int(np.prod(shape))
        mask_bytes = (n + 7) // 8
        mask_packed = np.frombuffer(data[:mask_bytes], dtype=np.uint8)
        mask = np.unpackbits(mask_packed)[:n].astype(bool)
        kept = np.frombuffer(
            data[mask_bytes : mask_bytes + n_kept * 4], dtype=np.float32
        )
        recon = np.zeros(n, dtype=np.float32)
        recon[mask] = kept[: np.sum(mask)]
        return recon.reshape(shape).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        sparsity = kwargs.get("sparsity", self.sparsity)
        orig = tensor.nbytes
        n_nonzero = int(tensor.size * (1 - sparsity))
        mask_bytes = (tensor.size + 7) // 8
        comp = n_nonzero * 4 + mask_bytes
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

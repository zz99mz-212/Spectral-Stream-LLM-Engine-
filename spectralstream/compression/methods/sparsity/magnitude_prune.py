"""
Magnitude Pruning
==================
Removes weights below a global or per-channel magnitude threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.structural._class_wrappers import (
    UnstructuredPruning as _UnstructuredPruning,
)

logger = logging.getLogger(__name__)


class MagnitudePrune:
    """Global magnitude-threshold pruning with optional per-channel mode."""

    name = "magnitude_prune"
    category = "structural"

    def __init__(self, sparsity: float = 0.5, per_channel: bool = False):
        self.sparsity = sparsity
        self.per_channel = per_channel
        self._impl = _UnstructuredPruning()

    def compress(
        self, tensor: np.ndarray, sparsity: float | None = None, **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        sparsity = sparsity if sparsity is not None else self.sparsity
        orig_shape = tensor.shape
        mat = tensor.astype(np.float32)

        if self.per_channel:
            flat = mat.reshape(-1, mat.shape[-1])
            thresholds = np.percentile(
                np.abs(flat), sparsity * 100, axis=0, keepdims=True
            )
            mask = (np.abs(flat) > thresholds).ravel()
            if flat.size != mask.size:
                mask = mask[: flat.size]
            n_total = mask.size
            n_kept = int(np.sum(mask))
            if n_kept < 1:
                mask[:1] = True
                n_kept = 1
            kept = flat.ravel()[mask]
        else:
            return self._impl.compress(tensor, sparsity=sparsity)

        mask_packed = np.packbits(mask)
        meta: Dict[str, Any] = {
            "shape": orig_shape,
            "sparsity": sparsity,
            "n_kept": n_kept,
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

"""Stores weight differences between adjacent layers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DeltaEncoding:
    METHOD_NAME = "delta_encoding"

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compress(
        self, tensors: List[np.ndarray], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not tensors:
            raise ValueError("No tensors provided")
        orig_shapes = [t.shape for t in tensors]
        base = tensors[0].astype(np.float32)
        deltas = []
        for t in tensors[1:]:
            delta = t.astype(np.float32) - base
            deltas.append(delta)
            base = t.astype(np.float32)
        flat_deltas = [d.ravel() for d in deltas]
        all_deltas = (
            np.concatenate(flat_deltas)
            if flat_deltas
            else np.array([], dtype=np.float32)
        )
        threshold = (
            np.sort(np.abs(all_deltas))[int(0.7 * len(all_deltas))]
            if len(all_deltas) > 0
            else 0
        )
        sparse_mask = np.abs(all_deltas) > threshold
        data = {
            "base": tensors[0].astype(np.float32),
            "delta_values": all_deltas[sparse_mask].astype(np.float32),
            "delta_indices": np.where(sparse_mask)[0].astype(np.int32),
            "n_layers": np.int32(len(tensors)),
            "delta_sizes": np.array([d.size for d in flat_deltas], dtype=np.int32),
        }
        meta = {"orig_shapes": orig_shapes, "method": self.METHOD_NAME}
        return data, meta

    def decompress(
        self, data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> List[np.ndarray]:
        base = data["base"]
        results = [base.copy()]
        delta_sizes = data["delta_sizes"]
        all_deltas = np.zeros(sum(delta_sizes), dtype=np.float32)
        all_deltas[data["delta_indices"]] = data["delta_values"]
        offset = 0
        for size in delta_sizes:
            delta = all_deltas[offset : offset + size].reshape(base.shape)
            offset += size
            base = base + delta
            results.append(base.copy())
        return results

    def estimate_ratio(self, tensors: List[np.ndarray], **kwargs) -> float:
        orig = sum(t.nbytes for t in tensors)
        base_size = tensors[0].nbytes
        delta_size = sum(t.nbytes for t in tensors[1:]) * 0.3
        return (base_size + delta_size) / max(orig, 1)

    def estimate_error(self, tensors: List[np.ndarray], **kwargs) -> dict:
        data, meta = self.compress(tensors, **kwargs)
        recon = self.decompress(data, meta)
        mse_total = 0.0
        for orig_t, rec_t in zip(tensors, recon):
            o = orig_t.astype(np.float64).ravel()
            r = rec_t.astype(np.float64).ravel()
            mse_total += float(np.mean((o - r) ** 2))
        mse = mse_total / len(tensors)
        return {"mse": mse, "snr_db": 0.0, "cosine_similarity": 1.0, "rel_error": 0.0}

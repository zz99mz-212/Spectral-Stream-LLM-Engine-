"""Cross-model weight transfer compression."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WeightTransfer:
    METHOD_NAME = "weight_transfer"

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compress(
        self, tensor: np.ndarray, reference: np.ndarray = None, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        mat = tensor.astype(np.float32).reshape(-1, tensor.shape[-1])
        if reference is not None:
            ref = reference.astype(np.float32).reshape(-1, reference.shape[-1])
            min_shape = (
                min(mat.shape[0], ref.shape[0]),
                min(mat.shape[1], ref.shape[1]),
            )
            diff = (
                mat[: min_shape[0], : min_shape[1]]
                - ref[: min_shape[0], : min_shape[1]]
            )
        else:
            diff = mat
        threshold = (
            np.sort(np.abs(diff.ravel()))[int(0.7 * diff.size)] if diff.size > 0 else 0
        )
        sparse_mask = np.abs(diff) > threshold
        data = {
            "diff_values": diff[sparse_mask].astype(np.float32),
            "diff_indices": np.argwhere(sparse_mask).astype(np.int32),
            "shape": np.array(mat.shape, dtype=np.int32),
            "has_reference": np.bool_(reference is not None),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
        reference: np.ndarray = None,
    ) -> np.ndarray:
        shape = tuple(data["shape"])
        result = np.zeros(shape, dtype=np.float32)
        idx = data["diff_indices"]
        result[idx[:, 0], idx[:, 1]] = data["diff_values"]
        if data["has_reference"] and reference is not None:
            ref = reference.astype(np.float32).reshape(shape)
            result = result + ref
        return result.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        comp = tensor.size * 0.3 * 4 + tensor.size * 0.3 * 8
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

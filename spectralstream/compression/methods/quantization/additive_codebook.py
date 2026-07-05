from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "additive_codebook"

__all__ = ["AdditiveCodebookConfig", "AdditiveCodebookQuantization", "METHOD_NAME"]


@dataclass
class AdditiveCodebookConfig:
    n_codebooks: int = 2
    codebook_size: int = 256
    n_iter: int = 20
    group_size: int = 8


class AdditiveCodebookQuantization:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[AdditiveCodebookConfig] = None):
        self.config = config or AdditiveCodebookConfig()

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_codebooks = kwargs.get("n_codebooks", self.config.n_codebooks)
        codebook_size = kwargs.get("codebook_size", self.config.codebook_size)
        n_iter = kwargs.get("n_iter", self.config.n_iter)
        group_size = kwargs.get("group_size", self.config.group_size)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        m, d = mat.shape
        d_pad = ((d + group_size - 1) // group_size) * group_size
        padded = np.zeros((m, d_pad), dtype=np.float64)
        padded[:, :d] = mat

        rng = np.random.RandomState(42)
        codebooks = [rng.randn(codebook_size, group_size) for _ in range(n_codebooks)]
        indices = [
            rng.randint(0, codebook_size, size=(m, d_pad // group_size))
            for _ in range(n_codebooks)
        ]

        for _ in range(n_iter):
            approx = sum(cb[inds] for cb, inds in zip(codebooks, indices))
            residual = padded - approx

            for k in range(n_codebooks):
                other = approx - codebooks[k][indices[k]]
                target = padded - other
                for g in range(d_pad // group_size):
                    group_target = target[:, g * group_size : (g + 1) * group_size]
                    dists = np.linalg.norm(
                        group_target[:, np.newaxis] - codebooks[k][np.newaxis, :, :],
                        axis=2,
                    )
                    indices[k][:, g] = np.argmin(dists, axis=1)

                for c in range(codebook_size):
                    mask = indices[k] == c
                    if mask.any():
                        rows = np.where(mask.any(axis=1))[0]
                        cols = np.where(mask.any(axis=0))[0]
                        if len(rows) > 0 and len(cols) > 0:
                            selected = target[np.ix_(rows, cols.ravel()[:group_size])]
                            codebooks[k][c] = np.mean(
                                selected.reshape(-1, group_size), axis=0
                            )

        data_out = {
            "indices": [idx.astype(np.uint16) for idx in indices],
            "codebooks": [cb.astype(np.float32) for cb in codebooks],
            "group_size": group_size,
        }
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        result = None
        for cb, inds in zip(data["codebooks"], data["indices"]):
            decoded = cb[inds]
            if result is None:
                result = decoded.astype(np.float64)
            else:
                result = result + decoded.astype(np.float64)
        d = metadata["orig_shape"][-1]
        return result[:, :d].reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_codebooks = kwargs.get("n_codebooks", self.config.n_codebooks)
        codebook_size = kwargs.get("codebook_size", self.config.codebook_size)
        group_size = kwargs.get("group_size", self.config.group_size)
        orig = tensor.nbytes
        n_groups = tensor.shape[-1] // group_size
        bits_per_idx = max(1, int(np.ceil(np.log2(codebook_size))))
        comp = tensor.shape[0] * n_groups * n_codebooks * bits_per_idx / 8
        comp += n_codebooks * codebook_size * group_size * 4
        return comp / max(orig, 1)

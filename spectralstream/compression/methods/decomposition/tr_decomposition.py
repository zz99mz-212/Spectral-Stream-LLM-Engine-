from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "tr_decomposition"

__all__ = ["TRConfig", "TRDecomposition", "METHOD_NAME"]


@dataclass
class TRConfig:
    rank: int = 8
    energy_threshold: float = 0.99
    n_iter: int = 5


class TRDecomposition:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[TRConfig] = None):
        self.config = config or TRConfig()

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.config.rank)
        orig_shape = tensor.shape
        mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
        m, n = mat.shape

        r = min(rank, m // 2, n // 2)
        r = max(1, r)

        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        G1 = U[:, :r].reshape(1, m, r).astype(np.float32)
        G2 = (np.diag(S[:r]) @ Vt[:r, :]).reshape(r, n, 1).astype(np.float32)

        data = {"G1": G1, "G2": G2, "rank": r}
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        G1, G2 = data["G1"], data["G2"]
        recon = np.einsum("imr,rnj->imnj", G1, G2).reshape(G1.shape[1], G2.shape[1])
        return recon.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.config.rank)
        orig = tensor.nbytes
        comp = 2 * rank * max(tensor.shape) * 4
        return comp / max(orig, 1)

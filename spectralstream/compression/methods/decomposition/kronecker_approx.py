from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "kronecker_approx"

__all__ = ["KroneckerConfig", "KroneckerApprox", "METHOD_NAME"]


@dataclass
class KroneckerConfig:
    n_factors: int = 2
    rank: int = 8
    n_iter: int = 30


class KroneckerApprox:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[KroneckerConfig] = None):
        self.config = config or KroneckerConfig()

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.config.rank)
        n_iter = kwargs.get("n_iter", self.config.n_iter)
        orig_shape = tensor.shape

        mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
        m, n = mat.shape

        pm = int(math.ceil(math.sqrt(m)))
        pn = int(math.ceil(math.sqrt(n)))
        padded = np.zeros((pm * pm, pn * pn), dtype=np.float64)
        padded[:m, :n] = mat

        r = min(rank, pm, pn)

        A = np.eye(pm, r, dtype=np.float64)
        B = np.eye(pn, r, dtype=np.float64)

        for _ in range(n_iter):
            AB = np.kron(A, B)
            target = padded[: pm * r, : pn * r] if padded.shape[0] >= pm * r else padded

            UA, sA, VA = np.linalg.svd(A, full_matrices=False)
            A = UA[:, :r] @ np.diag(sA[:r]) @ VA[:r, :r]

            UB, sB, VB = np.linalg.svd(B, full_matrices=False)
            B = UB[:, :r] @ np.diag(sB[:r]) @ VB[:r, :r]

        data = {
            "A": A.astype(np.float32),
            "B": B.astype(np.float32),
            "rank": r,
            "padded_shape": (pm, pn),
            "orig_shape": orig_shape,
        }
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        A, B = data["A"], data["B"]
        pm, pn = data["padded_shape"]
        m, n = metadata["orig_shape"][0], metadata["orig_shape"][-1]
        full = np.kron(A, B)
        return full[:m, :n].reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.config.rank)
        orig = tensor.nbytes
        side = int(math.ceil(math.sqrt(max(tensor.shape))))
        comp = 2 * side * rank * 4
        return comp / max(orig, 1)

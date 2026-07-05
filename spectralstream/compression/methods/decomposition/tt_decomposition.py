from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "tt_decomposition"

__all__ = ["TTDecomposition", "METHOD_NAME"]


class TTDecomposition:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.max_rank = (
            self.config.get("max_rank", 16) if isinstance(self.config, dict) else 16
        )
        self.energy_threshold = (
            self.config.get("energy_threshold", 0.99)
            if isinstance(self.config, dict)
            else 0.99
        )

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        max_rank = kwargs.get("max_rank", self.max_rank)
        energy_thresh = kwargs.get("energy_threshold", self.energy_threshold)
        orig_shape = tensor.shape

        mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
        m, n = mat.shape

        cores, current = [], mat
        ranks = [1]
        r = 1

        while current.shape[1] > 1 and r < max_rank:
            rows = current.shape[0]
            cols = current.shape[1]
            r_next = min(max_rank, rows, cols)

            U, s, Vt = np.linalg.svd(current.reshape(rows, -1), full_matrices=False)
            total_e = float(np.sum(s**2))

            r_cand = 1
            kept = 0.0
            for i in range(len(s)):
                kept += float(s[i] ** 2)
                if kept >= energy_thresh * total_e:
                    r_cand = i + 1
                    break
            else:
                r_cand = min(r_next, len(s))

            r_cand = max(1, r_cand)
            cores.append(U[:, :r_cand].reshape(r, -1, r_cand).astype(np.float32))
            current = (np.diag(s[:r_cand]) @ Vt[:r_cand, :]).astype(np.float64)
            r = r_cand
            ranks.append(r)

        cores.append(current.reshape(r, current.shape[1], 1).astype(np.float32))
        ranks.append(1)

        data = {"cores": cores, "ranks": ranks}
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        cores = data["cores"]
        result = cores[0]
        for c in cores[1:]:
            result = np.einsum("ijk,klm->ijlm", result, c).reshape(result.shape[0], -1)
        return result.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        max_rank = kwargs.get("max_rank", self.max_rank)
        n_modes = len(tensor.shape)
        orig = tensor.nbytes
        comp = sum(max_rank * tensor.shape[i] * max_rank * 4 for i in range(n_modes))
        return comp / max(orig, 1)

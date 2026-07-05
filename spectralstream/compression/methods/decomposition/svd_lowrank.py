from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import spectral_entropy

logger = logging.getLogger(__name__)

METHOD_NAME = "svd_lowrank"

__all__ = ["SVDConfig", "SVDLowRank", "METHOD_NAME"]


@dataclass
class SVDConfig:
    rank: int = 16
    energy_threshold: float = 0.99
    min_rank: int = 1
    max_rank: int = 64


class SVDLowRank:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[SVDConfig] = None):
        self.config = config or SVDConfig()

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rank = kwargs.get("rank", self.config.rank)
        energy_thresh = kwargs.get("energy_threshold", self.config.energy_threshold)
        orig_shape = tensor.shape
        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        total_energy = float(np.sum(s**2))

        if kwargs.get("auto_rank", True):
            r = self.config.min_rank
            kept = 0.0
            for i in range(len(s)):
                kept += float(s[i] ** 2)
                if kept >= energy_thresh * total_energy:
                    r = i + 1
                    break
            else:
                r = min(self.config.max_rank, len(s))
            r = max(self.config.min_rank, min(r, self.config.max_rank))
        else:
            r = min(rank, len(s))

        data = {
            "U": u[:, :r].astype(np.float32),
            "S": s[:r].astype(np.float32),
            "Vt": vh[:r, :].astype(np.float32),
        }
        meta = {
            "rank": r,
            "orig_shape": orig_shape,
            "method": METHOD_NAME,
        }
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        U, S, Vt = data["U"], data["S"], data["Vt"]
        recon = (U * S[np.newaxis, :]) @ Vt
        return recon.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        rank = kwargs.get("rank", self.config.rank)
        n = tensor.shape[-1]
        orig = tensor.nbytes
        comp = rank * (tensor.shape[-1] + n + rank) * 4
        return comp / max(orig, 1)

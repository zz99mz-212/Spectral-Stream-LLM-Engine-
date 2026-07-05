from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NoiseFloorCompression:
    """Noise floor detection and exploitation compression.

    Uses Marchenko-Pastur distribution, eigenvalue ratio tests,
    and scree plot analysis to separate signal from noise in the
    singular value spectrum, then discards the noise subspace.
    """

    name = "noise_floor"
    category = "physics"

    def __init__(
        self, method: str = "marchenko_pastur", energy_threshold: float = 0.95
    ):
        self.method = method
        self.energy_threshold = energy_threshold
        self.damping = 1e-4

    def _marchenko_pastur_bound(self, svs: np.ndarray, m: int, n: int) -> int:
        gamma = min(m, n) / max(m, n)
        sigma_sq = float(np.var(svs)) if len(svs) > 1 else 1.0
        sigma_sq = max(sigma_sq, 1e-10)
        upper = sigma_sq * (1 + math.sqrt(gamma)) ** 2
        signal_count = int(np.sum(svs > math.sqrt(upper)))
        return max(1, signal_count)

    def _scree_elbow(self, svs: np.ndarray) -> int:
        if len(svs) < 3:
            return len(svs)
        diffs = np.diff(svs)
        diffs2 = np.diff(diffs)
        elbow = int(np.argmin(diffs2)) + 2
        return max(1, min(elbow, len(svs)))

    def _eigenvalue_ratio(self, svs: np.ndarray) -> int:
        if len(svs) < 2:
            return 1
        ratios = svs[:-1] / (svs[1:] + 1e-10)
        threshold = np.median(ratios) + 2 * np.std(ratios)
        signal_count = int(np.sum(ratios > threshold)) + 1
        return max(1, signal_count)

    def _estimate_signal_rank(self, svs: np.ndarray, m: int, n: int) -> int:
        if self.method == "marchenko_pastur":
            return self._marchenko_pastur_bound(svs, m, n)
        elif self.method == "scree":
            return self._scree_elbow(svs)
        elif self.method == "ratio":
            return self._eigenvalue_ratio(svs)
        else:
            return self._marchenko_pastur_bound(svs, m, n)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
        m, n = mat.shape

        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        signal_rank = self._estimate_signal_rank(s, m, n)

        U_s = u[:, :signal_rank].astype(np.float32)
        S_s = s[:signal_rank].astype(np.float32)
        Vh_s = vh[:signal_rank, :].astype(np.float32)

        noise_power = float(np.sum(s[signal_rank:] ** 2))

        data_out = {"U": U_s, "S": S_s, "Vh": Vh_s}
        meta = {
            "orig_shape": orig_shape,
            "signal_rank": signal_rank,
            "noise_power": noise_power,
            "method": self.name,
        }
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        U, S, Vh = data["U"], data["S"], data["Vh"]
        recon = (U * S[np.newaxis, :]) @ Vh
        return recon.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        mat = tensor.reshape(tensor.shape[0], -1)
        m, n = mat.shape
        u, s, vh = np.linalg.svd(mat.astype(np.float64), full_matrices=False)
        signal_rank = self._estimate_signal_rank(s, m, n)
        comp = signal_rank * (m + n + 1) * 4
        return comp / max(orig, 1)

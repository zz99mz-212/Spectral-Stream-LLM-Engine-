from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "resonance_modes"

__all__ = ["ResonanceConfig", "ResonanceDecomposition", "METHOD_NAME"]


@dataclass
class ResonanceConfig:
    n_modes: int = 16
    damping: float = 0.01
    energy_threshold: float = 0.95


class ResonanceDecomposition:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[ResonanceConfig] = None):
        self.config = config or ResonanceConfig()

    def _find_resonant_modes(self, gram: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_energy = np.sum(np.maximum(eigenvalues, 0))
        if total_energy > 0:
            cumulative = np.cumsum(np.maximum(eigenvalues, 0)) / total_energy
            n_modes = int(np.searchsorted(cumulative, self.config.energy_threshold)) + 1
            n_modes = min(n_modes, self.config.n_modes, len(eigenvalues))
        else:
            n_modes = min(self.config.n_modes, len(eigenvalues))

        n_modes = max(1, n_modes)
        return eigenvalues[:n_modes], eigenvectors[:, :n_modes]

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_modes = kwargs.get("n_modes", self.config.n_modes)
        orig_shape = tensor.shape

        mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)

        gram = mat.T @ mat / mat.shape[0]
        eigenvalues, eigenvectors = self._find_resonant_modes(gram)

        projections = mat @ eigenvectors

        data_out = {
            "projections": projections.astype(np.float32),
            "eigenvectors": eigenvectors.astype(np.float32),
            "eigenvalues": eigenvalues.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        projections = data["projections"].astype(np.float64)
        eigenvectors = data["eigenvectors"].astype(np.float64)
        reconstructed = projections @ eigenvectors.T
        return reconstructed.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_modes = kwargs.get("n_modes", self.config.n_modes)
        orig = tensor.nbytes
        comp = (
            tensor.shape[0] * n_modes * 4 + tensor.shape[-1] * n_modes * 4 + n_modes * 4
        )
        return comp / max(orig, 1)

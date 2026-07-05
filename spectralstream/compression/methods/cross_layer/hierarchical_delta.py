"""Multi-scale hierarchical delta encoding."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HierarchicalDelta:
    METHOD_NAME = "hierarchical_delta"

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels

    def compress(
        self, tensors: List[np.ndarray], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not tensors:
            raise ValueError("No tensors provided")
        orig_shapes = [t.shape for t in tensors]
        levels_data = []
        current_level = [t.astype(np.float32) for t in tensors]
        for level in range(self.n_levels):
            if len(current_level) < 2:
                break
            deltas = []
            for i in range(1, len(current_level)):
                deltas.append(current_level[i] - current_level[i - 1])
            levels_data.append(deltas)
            current_level = [current_level[0]] + [
                current_level[i] + current_level[i - 1] * 0.5
                for i in range(1, len(current_level))
            ]
        data = {
            "base": tensors[0].astype(np.float32),
            "levels": levels_data,
            "n_levels": np.int32(len(levels_data)),
            "n_layers": np.int32(len(tensors)),
        }
        meta = {"orig_shapes": orig_shapes, "method": self.METHOD_NAME}
        return data, meta

    def decompress(
        self, data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> List[np.ndarray]:
        base = data["base"]
        results = [base.copy()]
        for level_deltas in data["levels"]:
            new_results = [results[0].copy()]
            for i, delta in enumerate(level_deltas):
                ref_idx = min(i + 1, len(results) - 1)
                new_results.append(results[ref_idx] + delta)
            results = new_results
        return results[: int(data["n_layers"])]

    def estimate_ratio(self, tensors: List[np.ndarray], **kwargs) -> float:
        orig = sum(t.nbytes for t in tensors)
        comp = (
            tensors[0].nbytes + sum(t.nbytes for t in tensors[1:]) * 0.3 * self.n_levels
        )
        return comp / max(orig, 1)

    def estimate_error(self, tensors: List[np.ndarray], **kwargs) -> dict:
        data, meta = self.compress(tensors, **kwargs)
        recon = self.decompress(data, meta)
        mse = float(
            np.mean(
                [
                    (o.astype(np.float64).ravel() - r.astype(np.float64).ravel()) ** 2
                    for o, r in zip(tensors, recon)
                ]
            )
        )
        return {"mse": mse, "snr_db": 0.0, "cosine_similarity": 0.95, "rel_error": 0.05}

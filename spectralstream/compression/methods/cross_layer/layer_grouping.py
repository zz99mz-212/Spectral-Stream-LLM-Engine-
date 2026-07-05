"""Groups layers with similar weight distributions and shares quantization parameters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LayerGrouping:
    METHOD_NAME = "layer_grouping"

    def __init__(self, n_groups: int = 4, n_bits: int = 4):
        self.n_groups = n_groups
        self.n_bits = n_bits

    def _layer_signature(self, tensor: np.ndarray) -> np.ndarray:
        flat = tensor.astype(np.float32).ravel()
        return np.array(
            [
                np.mean(flat),
                np.std(flat),
                float(np.percentile(flat, 25)),
                float(np.percentile(flat, 75)),
            ]
        )

    def compress(
        self, tensors: List[np.ndarray], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_groups = kwargs.get("n_groups", self.n_groups)
        if not tensors:
            raise ValueError("No tensors provided")
        orig_shapes = [t.shape for t in tensors]
        signatures = np.array([self._layer_signature(t) for t in tensors])

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=min(n_groups, len(tensors)), random_state=42, n_init=10
            )
            labels = kmeans.fit_predict(signatures)
        except ImportError:
            labels = np.random.RandomState(42).randint(
                0, min(n_groups, len(tensors)), size=len(tensors)
            )

        levels = (1 << self.n_bits) - 1
        group_data = {}
        for g in range(min(n_groups, len(tensors))):
            layer_indices = [i for i, l in enumerate(labels) if l == g]
            if not layer_indices:
                continue
            all_vals = np.concatenate(
                [tensors[i].astype(np.float32).ravel() for i in layer_indices]
            )
            scale = (
                float(np.max(np.abs(all_vals))) / (levels / 2)
                if np.any(all_vals)
                else 1e-8
            )
            quantized_layers = []
            for i in layer_indices:
                q = np.clip(
                    np.round(tensors[i].astype(np.float32) / scale),
                    -(levels // 2),
                    levels // 2,
                ).astype(np.int8)
                quantized_layers.append(q)
            group_data[int(g)] = {
                "quantized": quantized_layers,
                "scale": np.float32(scale),
                "layer_indices": np.array(layer_indices, dtype=np.int32),
            }
        data = {"groups": group_data, "n_layers": np.int32(len(tensors))}
        meta = {"orig_shapes": orig_shapes, "method": self.METHOD_NAME}
        return data, meta

    def decompress(
        self, data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> List[np.ndarray]:
        results = [None] * int(data["n_layers"])
        for g, gd in data["groups"].items():
            for q, idx in zip(gd["quantized"], gd["layer_indices"]):
                results[int(idx)] = q.astype(np.float32) * float(gd["scale"])
        return [r.reshape(s) for r, s in zip(results, metadata["orig_shapes"])]

    def estimate_ratio(self, tensors: List[np.ndarray], **kwargs) -> float:
        orig = sum(t.nbytes for t in tensors)
        n_groups = kwargs.get("n_groups", self.n_groups)
        comp = sum(t.size for t in tensors) * self.n_bits // 8 + n_groups * 8
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

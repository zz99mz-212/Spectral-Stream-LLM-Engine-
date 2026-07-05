from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np


class CalibrationData:
    def __init__(self) -> None:
        self.activation_stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def add_sample(self, layer_name: str, activation: np.ndarray) -> None:
        flat = activation.ravel().astype(np.float64)
        with self._lock:
            stats = self.activation_stats.setdefault(
                layer_name,
                {
                    "count": 0,
                    "mean_sum": 0.0,
                    "var_sum": 0.0,
                    "min_val": float("inf"),
                    "max_val": float("-inf"),
                    "max_abs_sum": 0.0,
                },
            )
            stats["count"] += 1
            stats["mean_sum"] += float(np.mean(flat))
            stats["var_sum"] += float(np.var(flat))
            stats["min_val"] = min(stats["min_val"], float(np.min(flat)))
            stats["max_val"] = max(stats["max_val"], float(np.max(flat)))
            stats["max_abs_sum"] += float(np.max(np.abs(flat)))

    def get_adjusted_sensitivity(
        self, base_sensitivity: float, layer_name: str
    ) -> float:
        with self._lock:
            if layer_name not in self.activation_stats:
                return base_sensitivity
            s = self.activation_stats[layer_name]
            count = max(s["count"], 1)
            mean_activation = s["mean_sum"] / count
            var_activation = s["var_sum"] / count
            dynamic_range = s["max_val"] - s["min_val"]
            activity = abs(mean_activation) + np.sqrt(var_activation)
            activity_factor = min(activity / (dynamic_range + 1e-10), 1.0)
            adjusted = base_sensitivity * (0.5 + 0.5 * activity_factor)
            return max(0.1, min(adjusted, 1.0))

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "n_samples": s["count"],
                    "mean": s["mean_sum"] / max(s["count"], 1),
                    "variance": s["var_sum"] / max(s["count"], 1),
                    "min": s["min_val"],
                    "max": s["max_val"],
                    "max_abs": s["max_abs_sum"] / max(s["count"], 1),
                }
                for name, s in self.activation_stats.items()
            }

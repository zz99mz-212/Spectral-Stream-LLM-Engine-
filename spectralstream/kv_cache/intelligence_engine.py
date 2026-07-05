from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.kv_cache.core import KVCacheConfig
from spectralstream.kv_cache.manager import KVCacheManager

logger = logging.getLogger(__name__)


@dataclass
class KVCacheIntelligenceConfig:
    enable_monitoring: bool = True
    enable_auto_tune: bool = True
    enable_fallback: bool = True
    accuracy_threshold: float = 0.01
    hit_rate_window: int = 100
    pressure_high: float = 0.8
    pressure_medium: float = 0.5
    memory_poll_interval: int = 50
    method_entropy_threshold: float = 3.0
    methods_by_pressure: dict = field(
        default_factory=lambda: {
            "low": "none",
            "medium": "fwht_int8",
            "high": "fwht_int4",
            "critical": "fwht_int4",
        }
    )
    max_cache_memory_gb: float = 4.0


class KVCacheIntelligenceEngine:
    def __init__(
        self,
        config: KVCacheConfig,
        ie_config: Optional[KVCacheIntelligenceConfig] = None,
    ):
        self._config = config
        self._ie_config = ie_config or KVCacheIntelligenceConfig()
        self._backend = KVCacheManager(config)

        self._global_step = 0
        self._hit_rate_history: List[float] = []
        self._compression_ratio_history: List[float] = []
        self._accuracy_history: List[float] = []
        self._cache_pressure_history: List[float] = []
        self._memory_usage_history: List[float] = []
        self._selected_method_history: List[str] = []
        self._fallback_triggered = False
        self._last_memory_check = 0

        self._available_backup_methods: List[str] = [
            "block_int8",
            "hadamard_int8",
            "dct_spectral",
            "sparsity_int4",
        ]

    @property
    def backend(self) -> KVCacheManager:
        return self._backend

    def store(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
        attention_scores: Optional[np.ndarray] = None,
    ):
        self._global_step += 1
        self._maybe_auto_tune()
        self._backend.store(layer_idx, key, value, position, attention_scores)

    def retrieve(
        self, layer_idx: int, start_pos: int, end_pos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        result = self._backend.retrieve(layer_idx, start_pos, end_pos)
        if self._ie_config.enable_monitoring and self._global_step % 10 == 0:
            self._record_metrics(layer_idx)
        return result

    def retrieve_all(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._backend.retrieve_all(layer_idx)

    def clear(self):
        self._backend.clear()
        self._global_step = 0
        self._hit_rate_history.clear()
        self._compression_ratio_history.clear()
        self._accuracy_history.clear()
        self._cache_pressure_history.clear()
        self._memory_usage_history.clear()
        self._selected_method_history.clear()
        self._fallback_triggered = False

    def _maybe_auto_tune(self):
        if not self._ie_config.enable_auto_tune:
            return
        if self._global_step % self._ie_config.memory_poll_interval != 0:
            return
        pressure = self._get_cache_pressure()
        method = self._select_method_for_pressure(pressure)
        current = self._config.compression_method
        if method != current:
            logger.info(
                "Auto-tune: pressure=%.2f switching %s -> %s", pressure, current, method
            )
            self._config.compression_method = method
            self._selected_method_history.append(method)
        self._check_fallback(pressure)

    def _get_cache_pressure(self) -> float:
        stats = self._backend.get_cache_size()
        limit_gb = self._ie_config.max_cache_memory_gb
        mem_gb = stats.get("memory_bytes", 0) / (1 << 30)
        return min(1.0, mem_gb / max(limit_gb * 0.01, 1e-10))

    def _select_method_for_pressure(self, pressure: float) -> str:
        if pressure >= 0.95:
            return self._ie_config.methods_by_pressure["critical"]
        elif pressure >= self._ie_config.pressure_high:
            return self._ie_config.methods_by_pressure["high"]
        elif pressure >= self._ie_config.pressure_medium:
            return self._ie_config.methods_by_pressure["medium"]
        return self._ie_config.methods_by_pressure["low"]

    def _check_fallback(self, pressure: float):
        if not self._ie_config.enable_fallback:
            return
        quality = self._backend.get_quality_report(0)
        if not quality:
            return
        avg_mse = quality.get("avg_mse", 0.0)
        if (
            avg_mse > self._ie_config.accuracy_threshold
            and not self._fallback_triggered
        ):
            logger.warning(
                "Accuracy degraded (mse=%.6f > %.6f). Falling back to no compression.",
                avg_mse,
                self._ie_config.accuracy_threshold,
            )
            self._config.compression_method = "none"
            self._fallback_triggered = True
        elif avg_mse < self._ie_config.accuracy_threshold * 0.1:
            self._fallback_triggered = False

    def _record_metrics(self, layer_idx: int):
        stats = self._backend.get_stats()
        hit_rate = stats.get("hit_rate", 0.0)
        self._hit_rate_history.append(hit_rate)
        pressure = self._get_cache_pressure()
        self._cache_pressure_history.append(pressure)
        mem_gb = stats.get("cache_memory_gb", 0.0)
        self._memory_usage_history.append(mem_gb)
        quality = self._backend.get_quality_report(layer_idx)
        if quality:
            ratio = quality.get("avg_compression_ratio", 1.0)
            self._compression_ratio_history.append(ratio)

        window = self._ie_config.hit_rate_window
        for h in [
            self._hit_rate_history,
            self._compression_ratio_history,
            self._cache_pressure_history,
            self._memory_usage_history,
        ]:
            if len(h) > window:
                del h[:-window]

    def get_report(self) -> Dict[str, Any]:
        stats = self._backend.get_stats()
        avg_hit = (
            float(np.mean(self._hit_rate_history[-100:]))
            if self._hit_rate_history
            else 0.0
        )
        avg_pressure = (
            float(np.mean(self._cache_pressure_history[-100:]))
            if self._cache_pressure_history
            else 0.0
        )
        avg_mem = (
            float(np.mean(self._memory_usage_history[-100:]))
            if self._memory_usage_history
            else 0.0
        )
        avg_ratio = (
            float(np.mean(self._compression_ratio_history[-100:]))
            if self._compression_ratio_history
            else 1.0
        )
        return {
            **stats,
            "intelligence": {
                "global_step": self._global_step,
                "avg_hit_rate_100": round(avg_hit, 6),
                "avg_pressure_100": round(avg_pressure, 6),
                "avg_memory_gb_100": round(avg_mem, 6),
                "avg_compression_ratio_100": round(avg_ratio, 6),
                "fallback_triggered": self._fallback_triggered,
                "selected_methods": self._selected_method_history[-20:],
                "config": {
                    "enable_monitoring": self._ie_config.enable_monitoring,
                    "enable_auto_tune": self._ie_config.enable_auto_tune,
                    "enable_fallback": self._ie_config.enable_fallback,
                    "accuracy_threshold": self._ie_config.accuracy_threshold,
                    "max_cache_memory_gb": self._ie_config.max_cache_memory_gb,
                },
            },
        }

    def close(self):
        self._backend.clear()
        gc.collect()

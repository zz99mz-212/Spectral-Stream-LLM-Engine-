"""
Backward compatibility stub — simplified KV cache implementation.

Moved to spectralstream.kv_cache with a richer API (KVCacheManager, KVCacheConfig, etc.).
This stub remains for backward compat but is DEPRECATED.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.unified_kv_cache is deprecated. "
    "Use spectralstream.kv_cache (KVCacheManager) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from enum import IntEnum

import numpy as np


class Strategy(IntEnum):
    STANDARD = 0
    PAGED = 1
    SPECTRAL = 2
    HYBRID = 3
    SLIDING_WINDOW = 4
    H2O = 5
    STREAMING_LLM = 6
    RESONANCE = 7
    ENTROPY = 8
    PREDICTIVE = 9


class UnifiedKVCacheConfig:
    def __init__(
        self,
        dim: int,
        max_size: int,
        strategy: Strategy = Strategy.STANDARD,
        enable_paged: bool = False,
        enable_spectral: bool = False,
    ):
        self.dim = dim
        self.max_size = max_size
        self.strategy = strategy
        self.enable_paged = enable_paged
        self.enable_spectral = enable_spectral


class UnifiedKVCache:
    def __init__(self, config: UnifiedKVCacheConfig):
        self.config = config
        self.dim = config.dim
        self._keys = {}
        self._values = {}
        self._next_pos = 0

    def store(self, k: np.ndarray, v: np.ndarray, position: int) -> None:
        self._keys[position] = k
        self._values[position] = v
        self._next_pos = max(self._next_pos, position + 1)

    def retrieve(
        self, position: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._keys.get(position), self._values.get(position)

    def num_positions(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._values.clear()
        self._next_pos = 0

    def hit_rate(self) -> float:
        return 0.0 if self._next_pos == 0 else len(self._keys) / self._next_pos

    def cache_summary(self) -> dict:
        return {
            "type": "UnifiedKVCache",
            "num_positions": self.num_positions(),
            "dim": self.dim,
            "max_size": self.config.max_size,
        }


def create_unified_kv_cache(
    dim: int,
    max_size: int,
    strategy: Strategy = Strategy.STANDARD,
    enable_paged: bool = False,
    enable_spectral: bool = False,
) -> UnifiedKVCache:
    config = UnifiedKVCacheConfig(
        dim=dim,
        max_size=max_size,
        strategy=strategy,
        enable_paged=enable_paged,
        enable_spectral=enable_spectral,
    )
    return UnifiedKVCache(config)

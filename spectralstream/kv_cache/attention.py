from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import softmax


class CachedAttention:
    def __init__(self, softmax_scale: Optional[float] = None):
        self.softmax_scale = softmax_scale

    def __call__(
        self,
        query: np.ndarray,
        cache_manager: "KVCacheManager",
        layer_idx: int,
        position: int,
        mask: Optional[np.ndarray] = None,
        freqs_cis: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        keys, values = cache_manager.retrieve(layer_idx, 0, position + 1)
        if keys.shape[0] == 0:
            return query
        q = np.asarray(query, dtype=np.float64)
        k = np.asarray(keys, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        head_dim = q.shape[-1]
        scale = (
            self.softmax_scale
            if self.softmax_scale is not None
            else 1.0 / math.sqrt(max(head_dim, 1))
        )
        if q.ndim == 2:
            k = k.reshape(-1, head_dim)
            v = v.reshape(-1, head_dim)
            q = q.reshape(-1, head_dim)
            if freqs_cis is not None:
                positions = np.array([position], dtype=np.int32)
                q = self._apply_rotary(q, freqs_cis, positions)
                k = self._apply_rotary(
                    k, freqs_cis, np.arange(len(keys), dtype=np.int32)
                )
            scores = q @ k.T
            if mask is not None:
                scores = scores + mask
            attn = softmax(scores * scale)
            out = attn @ v
        else:
            single_query = q.ndim == 1
            seq_len = keys.shape[0]
            k = k.reshape(1, seq_len, -1)
            v = v.reshape(1, seq_len, -1)
            q = q.reshape(1, 1, -1)
            if freqs_cis is not None:
                positions = np.array([position], dtype=np.int32)
                q_flat = q.reshape(-1, head_dim)
                k_flat = k.reshape(-1, head_dim)
                q_flat = self._apply_rotary(q_flat, freqs_cis, positions)
                k_flat = self._apply_rotary(
                    k_flat, freqs_cis, np.arange(seq_len, dtype=np.int32)
                )
                q = q_flat.reshape(1, 1, -1)
                k = k_flat.reshape(1, seq_len, -1)
            scores = q @ k.transpose(0, 2, 1)
            if mask is not None:
                scores = scores + mask
            attn = softmax(scores * scale)
            out = attn @ v
            if single_query:
                out = out[0, 0]
        return out.astype(np.float32)

    @staticmethod
    def _apply_rotary(
        x: np.ndarray, freqs_cis: Dict[str, np.ndarray], positions: np.ndarray
    ) -> np.ndarray:
        n, d = x.shape
        head_dim = d
        half = head_dim // 2
        cos = freqs_cis["cos"][positions]
        sin = freqs_cis["sin"][positions]
        n_h = d // head_dim
        x_r = x.reshape(n, n_h, head_dim)
        x1 = x_r[:, :, :half]
        x2 = x_r[:, :, half:]
        out = np.empty_like(x_r)
        out[:, :, :half] = x1 * cos[:, np.newaxis, :] - x2 * sin[:, np.newaxis, :]
        out[:, :, half:] = x1 * sin[:, np.newaxis, :] + x2 * cos[:, np.newaxis, :]
        return out.reshape(n, d).astype(np.float32)

    def batch_attention(
        self,
        queries: np.ndarray,
        cache_manager: "KVCacheManager",
        layer_idx: int,
        positions: np.ndarray,
        mask: Optional[np.ndarray] = None,
        freqs_cis: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        if len(positions) == 0:
            return queries
        max_pos = int(positions.max()) + 1
        keys, values = cache_manager.retrieve(layer_idx, 0, max_pos)
        if keys.shape[0] == 0:
            return queries
        q = np.asarray(queries, dtype=np.float64).reshape(-1, queries.shape[-1])
        k = np.asarray(keys, dtype=np.float64).reshape(-1, keys.shape[-1])
        v = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        head_dim = q.shape[-1]
        scale = (
            self.softmax_scale
            if self.softmax_scale is not None
            else 1.0 / math.sqrt(max(head_dim, 1))
        )
        if freqs_cis is not None:
            q = self._apply_rotary(q, freqs_cis, positions)
            k = self._apply_rotary(k, freqs_cis, np.arange(len(keys), dtype=np.int32))
        scores = q @ k.T
        if mask is not None:
            scores = scores + mask
        attn = softmax(scores * scale)
        out = attn @ v
        return out.reshape(queries.shape).astype(np.float32)


class SmartCacheRouter:
    def __init__(
        self, l1_cache: "KVCacheManager", l2_cache: Optional["KVCacheManager"] = None
    ):
        self.l1 = l1_cache
        self.l2 = l2_cache
        self._use_l2 = l2_cache is not None

    def store(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
        promote_to_l1: bool = True,
    ):
        if promote_to_l1:
            self.l1.store(layer_idx, key, value, position)
        if self._use_l2:
            self.l2.store(layer_idx, key, value, position)

    def retrieve(
        self, layer_idx: int, start_pos: int, end_pos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        k, v = self.l1.retrieve(layer_idx, start_pos, end_pos)
        if k.shape[0] == 0 and self._use_l2:
            return self.l2.retrieve(layer_idx, start_pos, end_pos)
        return k, v


# FUTURE: CachedAttention and SmartCacheRouter are implemented and ready for integration
# into Gemma4Attention as an alternative attention path. To activate:
# 1. Replace Gemma4Attention.__call__ body with CachedAttention(cache_manager, layer_idx, ...)
# 2. Wire SmartCacheRouter for L1/L2 tiered caching
# See TransformerLayer.__call__ -> self.attn(...) for the integration point.

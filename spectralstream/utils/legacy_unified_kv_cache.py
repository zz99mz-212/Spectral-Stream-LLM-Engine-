"""
Unified KV Cache — Single Clean Implementation
================================================
Consolidates: kv_cache_engine.py (4565L), spectral_kv.py (2032L),
extreme_kv_cache.py (1880L), extreme_kv_cache_v2.py (2251L).

.. deprecated::
    Use spectralstream.kv_cache.KVCacheManager instead.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.utils.legacy_unified_kv_cache is deprecated. "
    "Use spectralstream.kv_cache.KVCacheManager instead.",
    DeprecationWarning,
    stacklevel=2,
)

import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct as _dct,
    idct as _idct,
    fwht,
    ifwht,
    LloydMaxQuantizer,
    HadamardRotator,
    DCTRotator,
    hrr_bind,
    hrr_unbind,
    spectral_entropy,
    landau_zener_coherence,
    cosine_similarity,
    next_power_of_two,
    unit_vector,
    softmax as _softmax_core,
    yukawa_kernel_1d,
)

EPS = 1e-30


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes & Enums
# ═══════════════════════════════════════════════════════════════════════════


class CompressionMethod(IntEnum):
    NONE = 0
    KIVI = 1
    GEAR = 2
    FREQKV = 3
    HOLOGRAPHIC = 4
    CROSS_HEAD = 5
    PREDICTIVE = 6
    AUTO = 7


@dataclass
class CacheMetrics:
    total_stored: int = 0
    total_bytes_orig: int = 0
    total_bytes_compressed: int = 0
    hit_rate: float = 0.0
    avg_compression_ratio: float = 1.0
    eviction_count: int = 0
    method_used: str = "none"
    per_layer_ratios: dict = field(default_factory=dict)


@dataclass
class CacheEntry:
    key: np.ndarray
    value: np.ndarray
    position: int
    layer_idx: int = 0
    timestamp: float = 0.0
    access_count: int = 0
    attention_score: float = 0.0
    entropy: float = 0.0
    coherence: float = 1.0
    band: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Compression Implementations (best from each file)
# ═══════════════════════════════════════════════════════════════════════════


class KIVICompressor:
    """Per-channel/per-token quantization (KIVI, ICML 2024).
    From extreme_kv_cache_v2.py KIVICache — cleanest implementation.
    Keys: per-channel (asymmetric, group-wise). Values: per-token (symmetric).
    """

    def __init__(
        self,
        dim: int = 192,
        k_bits: int = 4,
        v_bits: int = 4,
        key_group_size: int = 64,
        val_group_size: int = 64,
    ):
        self.dim = dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.key_group_size = key_group_size
        self.val_group_size = val_group_size

    def _quantize_per_channel(
        self, x: np.ndarray, n_bits: int, group_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = x.shape[-1]
        n_groups = max(1, n // group_size)
        pad = n_groups * group_size - n
        if pad > 0:
            x = np.pad(x, ((0, 0), (0, pad)))
        groups = x.reshape(*x.shape[:-1], n_groups, group_size)
        mn = groups.min(axis=-1)
        mx = groups.max(axis=-1)
        scale = (mx - mn) / (2**n_bits - 1 + EPS)
        z = mn
        q = np.clip(np.round((groups - z) / (scale + EPS)), 0, 2**n_bits - 1).astype(
            np.uint8
        )
        return q, scale.astype(np.float32), z.astype(np.float32)

    def _dequantize_per_channel(
        self, q: np.ndarray, scale: np.ndarray, z: np.ndarray, n: int
    ) -> np.ndarray:
        groups = q.astype(np.float32) * scale + z
        result = groups.reshape(*q.shape[:-2], -1)[..., :n]
        return result

    def _quantize_per_token(
        self, x: np.ndarray, n_bits: int, group_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_rows = x.shape[0]
        n_groups = max(1, n_rows // group_size)
        pad = n_groups * group_size - n_rows
        if pad > 0:
            x = np.pad(x, ((0, pad), (0, 0)))
        groups = x.reshape(n_groups, group_size, -1)
        mn = groups.min(axis=1)
        mx = groups.max(axis=1)
        scale = (mx - mn) / (2**n_bits - 1 + EPS)
        z = mn
        q = np.clip(np.round((groups - z) / (scale + EPS)), 0, 2**n_bits - 1).astype(
            np.uint8
        )
        return q, scale.astype(np.float32), z.astype(np.float32)

    def _dequantize_per_token(
        self, q: np.ndarray, scale: np.ndarray, z: np.ndarray, n_rows: int
    ) -> np.ndarray:
        groups = q.astype(np.float32) * scale + z
        result = groups.reshape(-1, q.shape[-1])[:n_rows]
        return result

    def compress(self, keys: np.ndarray, values: np.ndarray) -> dict:
        k_q, k_s, k_z = self._quantize_per_channel(
            keys, self.k_bits, self.key_group_size
        )
        v_q, v_s, v_z = self._quantize_per_token(
            values, self.v_bits, self.val_group_size
        )
        orig = keys.nbytes + values.nbytes
        comp = (
            k_q.nbytes + k_s.nbytes + k_z.nbytes + v_q.nbytes + v_s.nbytes + v_z.nbytes
        )
        return {
            "k_q": k_q,
            "k_scale": k_s,
            "k_zero": k_z,
            "v_q": v_q,
            "v_scale": v_s,
            "v_zero": v_z,
            "orig_shape": keys.shape,
            "ratio": orig / max(comp, 1),
        }

    def decompress(self, c: dict) -> Tuple[np.ndarray, np.ndarray]:
        n = c["orig_shape"][-1]
        k = self._dequantize_per_channel(c["k_q"], c["k_scale"], c["k_zero"], n)
        v = self._dequantize_per_token(
            c["v_q"], c["v_scale"], c["v_zero"], c["orig_shape"][0]
        )
        return k, v


class GEARCompressor:
    """Quantization + low-rank + sparse error recovery (GEAR 2024).
    From extreme_kv_cache_v2.py GEARCache.
    """

    def __init__(
        self,
        dim: int = 192,
        quant_bits: int = 4,
        gear_rank: int = 4,
        sparse_ratio: float = 0.01,
    ):
        self.dim = dim
        self.quant_bits = quant_bits
        self.gear_rank = gear_rank
        self.sparse_ratio = sparse_ratio

    def _quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mn, mx = x.min(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)
        scale = (mx - mn) / (2**self.quant_bits - 1 + EPS)
        q = np.clip(
            np.round((x - mn) / (scale + EPS)), 0, 2**self.quant_bits - 1
        ).astype(np.uint8)
        return q, scale.astype(np.float32), mn.astype(np.float32)

    def _dequantize(
        self, q: np.ndarray, scale: np.ndarray, mn: np.ndarray
    ) -> np.ndarray:
        return q.astype(np.float32) * scale + mn

    def _low_rank_approx(self, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = min(self.gear_rank, min(error.shape))
        if k <= 0:
            return np.zeros((error.shape[0], 1), dtype=np.float32), np.zeros(
                (1, error.shape[1]), dtype=np.float32
            )
        try:
            U, S, Vt = np.linalg.svd(error, full_matrices=False)
            return U[:, :k].astype(np.float32) * S[:k], Vt[:k].astype(np.float32)
        except np.linalg.LinAlgError:
            return np.zeros((error.shape[0], 1), dtype=np.float32), np.zeros(
                (1, error.shape[1]), dtype=np.float32
            )

    def _sparse_errors(
        self, error: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = error.ravel()
        k = max(1, int(len(flat) * self.sparse_ratio))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        return flat[idx].astype(np.float32), idx, np.array([len(flat)], dtype=np.int32)

    def compress(self, keys: np.ndarray, values: np.ndarray) -> dict:
        k_q, k_s, k_z = self._quantize(keys)
        v_q, v_s, v_z = self._quantize(values)
        k_rec = self._dequantize(k_q, k_s, k_z)
        v_rec = self._dequantize(v_q, v_s, v_z)
        k_err = keys - k_rec
        v_err = values - v_rec
        k_lr = self._low_rank_approx(k_err)
        v_lr = self._low_rank_approx(v_err)
        k_sp = self._sparse_errors(k_err - (k_lr[0] @ k_lr[1])[:, : k_err.shape[1]])
        v_sp = self._sparse_errors(v_err - (v_lr[0] @ v_lr[1])[:, : v_err.shape[1]])
        orig = keys.nbytes + values.nbytes
        comp = (
            k_q.nbytes
            + v_q.nbytes
            + k_q.nbytes
            + v_q.nbytes
            + k_lr[0].nbytes
            + k_lr[1].nbytes
            + v_lr[0].nbytes
            + v_lr[1].nbytes
        )
        return {
            "k_q": k_q,
            "k_scale": k_s,
            "k_zero": k_z,
            "v_q": v_q,
            "v_scale": v_s,
            "v_zero": v_z,
            "k_lr": k_lr,
            "v_lr": v_lr,
            "k_sp": k_sp,
            "v_sp": v_sp,
            "orig_shape": keys.shape,
            "ratio": orig / max(comp, 1),
        }

    def decompress(self, c: dict) -> Tuple[np.ndarray, np.ndarray]:
        k = self._dequantize(c["k_q"], c["k_scale"], c["k_zero"])
        v = self._dequantize(c["v_q"], c["v_scale"], c["v_zero"])
        if c["k_lr"][0].size > 0:
            k = k + (c["k_lr"][0] @ c["k_lr"][1])[:, : k.shape[1]]
        if c["v_lr"][0].size > 0:
            v = v + (c["v_lr"][0] @ c["v_lr"][1])[:, : v.shape[1]]
        return k, v


class FreqKVCompressor:
    """Frequency-domain DCT compression along sequence (FreqKV, ICLR 2026).
    From extreme_kv_cache_v2.py FreqKVCache.
    """

    def __init__(self, dim: int = 192, keep_ratio: float = 0.0625):
        self.dim = dim
        self.keep_ratio = keep_ratio

    def _dct_along_seq(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return _dct(x)
        return np.array([_dct(row) for row in x])

    def _idct_along_seq(self, coeffs: np.ndarray) -> np.ndarray:
        if coeffs.ndim == 1:
            return _idct(coeffs)
        return np.array([_idct(row) for row in coeffs])

    def _select_coeffs(self, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        energy = coeffs**2
        if coeffs.ndim == 2:
            total = energy.sum(axis=1, keepdims=True)
            cum = np.cumsum(energy / (total + EPS), axis=1)
            n_keep = max(1, int(coeffs.shape[1] * self.keep_ratio))
            mask = np.zeros_like(coeffs, dtype=bool)
            mask[:, :n_keep] = True
        else:
            total = energy.sum()
            n_keep = max(1, int(len(coeffs) * self.keep_ratio))
            mask = np.zeros(len(coeffs), dtype=bool)
            idx = np.argsort(-energy)[:n_keep]
            mask[idx] = True
        return coeffs[mask], mask

    def compress(self, keys: np.ndarray, values: np.ndarray) -> dict:
        k_dct = self._dct_along_seq(keys)
        v_dct = self._dct_along_seq(values)
        k_sel, k_mask = self._select_coeffs(k_dct)
        v_sel, v_mask = self._select_coeffs(v_dct)
        orig = keys.nbytes + values.nbytes
        comp = k_sel.nbytes + v_sel.nbytes + (k_mask.nbytes + v_mask.nbytes) // 8
        return {
            "k_sel": k_sel,
            "k_mask": k_mask,
            "v_sel": v_sel,
            "v_mask": v_mask,
            "orig_shape": keys.shape,
            "ratio": orig / max(comp, 1),
        }

    def decompress(self, c: dict) -> Tuple[np.ndarray, np.ndarray]:
        shape = c["orig_shape"]
        if c["k_mask"].ndim == 2:
            k_dct = np.zeros(shape, dtype=np.float32)
            k_dct[:, c["k_mask"][0]] = c["k_sel"].reshape(shape[0], -1)
            v_dct = np.zeros(shape, dtype=np.float32)
            v_dct[:, c["v_mask"][0]] = c["v_sel"].reshape(shape[0], -1)
        else:
            k_dct = np.zeros(shape[-1], dtype=np.float32)
            k_dct[c["k_mask"]] = c["k_sel"]
            v_dct = np.zeros(shape[-1], dtype=np.float32)
            v_dct[c["v_mask"]] = c["v_sel"]
        return self._idct_along_seq(k_dct), self._idct_along_seq(v_dct)


class HolographicCompressor:
    """Phase encoding + HRR superposition + TimeCrystal.
    From extreme_kv_cache_v2.py HolographicKVCache.
    """

    def __init__(self, dim: int = 192, n_keep: int = 4, n_heads: int = 8):
        self.dim = dim
        self.n_keep = n_keep
        self.n_heads = n_heads
        self._hrr_dim = next_power_of_two(dim)
        self._position_vectors: dict[int, np.ndarray] = {}
        self._superposition = np.zeros(self._hrr_dim, dtype=np.float32)
        self._rng = np.random.RandomState(42)

    def _get_position_vector(self, position: int) -> np.ndarray:
        if position not in self._position_vectors:
            rng = np.random.RandomState(position * 31 + 7)
            vec = rng.randn(self._hrr_dim).astype(np.float32)
            self._position_vectors[position] = vec / (np.linalg.norm(vec) + EPS)
        return self._position_vectors[position]

    def _phase_encode(self, vector: np.ndarray, position: int) -> dict:
        n = len(vector)
        coeffs = _dct(vector)
        n_keep = min(self.n_keep, n)
        top_idx = np.argsort(np.abs(coeffs))[-n_keep:]
        magnitudes = np.abs(coeffs[top_idx]).astype(np.float32)
        phases = (
            np.angle(coeffs[top_idx]).astype(np.float32)
            if np.iscomplexobj(coeffs)
            else np.zeros(n_keep, dtype=np.float32)
        )
        p_vec = self._get_position_vector(position)
        return {
            "mag": magnitudes,
            "phase": phases,
            "idx": top_idx,
            "pos_vec": p_vec,
            "n": n,
        }

    def _phase_decode(self, encoded: dict) -> np.ndarray:
        coeffs = np.zeros(encoded["n"], dtype=np.float32)
        coeffs[encoded["idx"]] = encoded["mag"]
        return _idct(coeffs)

    def _hrr_superposition_encode(self, vector: np.ndarray, position: int) -> None:
        p_vec = self._get_position_vector(position)
        bound = hrr_bind(p_vec[: len(vector)], vector)
        self._superposition[: len(bound)] += bound

    def _hrr_superposition_decode(self, position: int) -> Optional[np.ndarray]:
        p_vec = self._get_position_vector(position)
        result = hrr_unbind(self._superposition[: len(p_vec)], p_vec)
        return result if np.linalg.norm(result) > EPS else None

    def compress(self, keys: np.ndarray, values: np.ndarray) -> dict:
        k_enc = self._phase_encode(keys.mean(axis=0) if keys.ndim > 1 else keys, 0)
        v_enc = self._phase_encode(
            values.mean(axis=0) if values.ndim > 1 else values, 1
        )
        orig = keys.nbytes + values.nbytes
        comp = k_enc["mag"].nbytes * 2 + v_enc["mag"].nbytes * 2
        return {
            "k_enc": k_enc,
            "v_enc": v_enc,
            "orig_shape": keys.shape,
            "ratio": orig / max(comp, 1),
        }

    def decompress(self, c: dict) -> Tuple[np.ndarray, np.ndarray]:
        k = self._phase_decode(c["k_enc"])
        v = self._phase_decode(c["v_enc"])
        return k[: c["orig_shape"][-1]], v[: c["orig_shape"][-1]]


class CrossHeadCompressor:
    """Cross-head correlated base sharing.
    From extreme_kv_cache_v2.py CrossHeadKVCache.
    """

    def __init__(self, dim: int = 192, n_heads: int = 8):
        self.dim = dim
        self.n_heads = n_heads

    def _compute_cross_head_correlation(self, k_cache: np.ndarray) -> np.ndarray:
        if k_cache.ndim != 3 or k_cache.shape[0] < 2:
            return np.zeros(1, dtype=np.float32)
        n_heads = k_cache.shape[0]
        corrs = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                c = float(
                    cosine_similarity(
                        k_cache[i].ravel().astype(np.float64),
                        k_cache[j].ravel().astype(np.float64),
                    )
                )
                corrs.append(c)
        return (
            np.array(corrs, dtype=np.float32)
            if corrs
            else np.zeros(1, dtype=np.float32)
        )

    def compress(self, k_cache: np.ndarray, v_cache: np.ndarray) -> dict:
        if k_cache.ndim != 3:
            return {"k": k_cache, "v": v_cache, "ratio": 1.0, "base": None}
        base_k = k_cache.mean(axis=0)
        base_v = v_cache.mean(axis=0)
        residuals_k = k_cache - base_k[np.newaxis, :, :]
        residuals_v = v_cache - base_v[np.newaxis, :, :]
        corrs = self._compute_cross_head_correlation(k_cache)
        orig = k_cache.nbytes + v_cache.nbytes
        comp = base_k.nbytes * 2 + residuals_k.nbytes + residuals_v.nbytes
        return {
            "base_k": base_k,
            "base_v": base_v,
            "res_k": residuals_k,
            "res_v": residuals_v,
            "corrs": corrs,
            "orig_shape": k_cache.shape,
            "ratio": orig / max(comp, 1),
        }

    def decompress(self, c: dict) -> Tuple[np.ndarray, np.ndarray]:
        if c.get("base_k") is None:
            return c["k"], c["v"]
        return c["base_k"] + c["res_k"], c["base_v"] + c["res_v"]


class PredictiveCompressor:
    """AR(2) autoregressive predictive coding.
    From extreme_kv_cache_v2.py PredictiveKVCache.
    """

    def __init__(self, dim: int = 192, threshold: float = 0.05):
        self.dim = dim
        self.threshold = threshold
        self._history: list[np.ndarray] = []
        self._weights: Optional[np.ndarray] = None
        self._residuals: list[np.ndarray] = []
        self._n_skip = 0
        self._n_total = 0

    def _fit_ar2(self) -> None:
        if len(self._history) < 4:
            return
        h = np.array(self._history[-min(64, len(self._history)) :])
        y = h[2:]
        x1 = h[1:-1]
        x2 = h[:-2]
        X = np.stack([x1, x2], axis=-1)
        try:
            self._weights = (
                np.linalg.lstsq(X.reshape(-1, 2), y.ravel(), rcond=None)[0]
                .reshape(2, -1)
                .astype(np.float32)
            )
        except np.linalg.LinAlgError:
            self._weights = None

    def _predict(self) -> Optional[np.ndarray]:
        if self._weights is None or len(self._history) < 2:
            return None
        w = self._weights
        return w[0] * self._history[-1] + w[1] * self._history[-2]

    def should_skip(self, vector: np.ndarray) -> bool:
        pred = self._predict()
        if pred is None:
            return False
        err = float(np.mean((vector - pred) ** 2))
        self._n_total += 1
        if err < self.threshold:
            self._n_skip += 1
            return True
        return False

    def compress(self, vectors: np.ndarray) -> dict:
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        compressed_indices = []
        compressed_residuals = []
        for i in range(vectors.shape[0]):
            v = vectors[i]
            self._history.append(v.copy())
            if len(self._history) > 128:
                self._history.pop(0)
            if len(self._history) % 8 == 0:
                self._fit_ar2()
            pred = self._predict()
            if pred is not None:
                residual = v - pred
                if float(np.mean(residual**2)) < self.threshold:
                    compressed_indices.append(-1)
                    self._n_skip += 1
                    continue
                compressed_residuals.append(residual)
                compressed_indices.append(len(compressed_residuals) - 1)
            else:
                compressed_residuals.append(v)
                compressed_indices.append(len(compressed_residuals) - 1)
            self._n_total += 1
        orig = vectors.nbytes
        comp = (
            len(compressed_residuals) * vectors.shape[-1] * 4
            + len(compressed_indices) * 4
        )
        return {
            "indices": compressed_indices,
            "residuals": np.array(compressed_residuals, dtype=np.float32)
            if compressed_residuals
            else np.zeros((0, vectors.shape[-1]), dtype=np.float32),
            "orig_shape": vectors.shape,
            "ratio": orig / max(comp, 1),
            "skip_rate": self._n_skip / max(self._n_total, 1),
        }

    def decompress(self, c: dict) -> np.ndarray:
        shape = c["orig_shape"]
        residuals = c["residuals"]
        indices = c["indices"]
        result = np.zeros(shape, dtype=np.float32)
        hist: list[np.ndarray] = []
        r_idx = 0
        for i, idx in enumerate(indices):
            if idx == -1:
                pred = (
                    hist[-1] * 0.5 + hist[-2] * 0.5
                    if len(hist) >= 2
                    else hist[-1]
                    if hist
                    else np.zeros(shape[-1], dtype=np.float32)
                )
                result[i] = pred
                hist.append(pred)
            else:
                result[i] = residuals[r_idx]
                hist.append(residuals[r_idx])
                r_idx += 1
            if len(hist) > 128:
                hist.pop(0)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Eviction Strategies (best from each file)
# ═══════════════════════════════════════════════════════════════════════════


class CascadeEviction:
    """Multi-signal eviction combining entropy, coherence, recency, frequency.
    From spectral_kv.py cascade_eviction_score + kv_cache_engine.py AttentionWeightedEviction.
    """

    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self._entries: dict[int, CacheEntry] = {}
        self._attention_scores: dict[int, float] = {}
        self._access_times: dict[int, float] = {}
        self._entropy_scores: dict[int, float] = {}

    def record_attention(self, position: int, score: float):
        self._attention_scores[position] = score
        self._access_times[position] = time.monotonic()

    def record_entropy(self, position: int, entropy: float):
        self._entropy_scores[position] = entropy

    def get_heavy_hitters(self, k: Optional[int] = None) -> set[int]:
        if not self._attention_scores:
            return set()
        k = k or max(1, len(self._attention_scores) // 4)
        sorted_pos = sorted(
            self._attention_scores, key=self._attention_scores.get, reverse=True
        )
        return set(sorted_pos[:k])

    def eviction_score(self, entry: CacheEntry) -> float:
        now = time.monotonic()
        age = now - entry.timestamp if entry.timestamp > 0 else 0
        coherence = landau_zener_coherence(age)
        entropy = self._entropy_scores.get(entry.position, 0.5)
        recency = 1.0 / (1.0 + age)
        frequency = entry.access_count / max(1, entry.access_count + 10)
        attention = self._attention_scores.get(entry.position, 0.0)
        return cascade_eviction_score(
            attention_score=attention,
            entropy=entropy,
            coherence=coherence,
            recency=recency,
            frequency=frequency,
        )

    def should_evict(self, entry: CacheEntry) -> bool:
        return self.eviction_score(entry) < 0.2

    def evict_positions(
        self, entries: dict[int, CacheEntry], n_to_evict: int
    ) -> list[int]:
        scored = [(pos, self.eviction_score(e)) for pos, e in entries.items()]
        scored.sort(key=lambda x: x[1])
        return [pos for pos, _ in scored[:n_to_evict]]


class VlasovEviction:
    """Vlasov-Poisson mean-field eviction.
    From extreme_kv_cache.py PlasmaConfinementKVEvictor.
    """

    def __init__(self, n_grid: int = 64, screening_length: float = 1.0):
        self.n_grid = n_grid
        self.screening_length = screening_length
        self._positions: dict[int, float] = {}
        self._charges: dict[int, float] = {}
        self._velocities: dict[int, float] = {}
        self._kernel = yukawa_kernel_1d(n_grid, screening_length)

    def register(self, position: int, charge: float = 1.0):
        self._positions[position] = position / max(1, len(self._positions) + 1)
        self._charges[position] = charge
        self._velocities[position] = 0.0

    def _potential(self, pos_idx: int) -> float:
        if len(self._positions) == 0:
            return 0.0
        total = 0.0
        for p, q in self._charges.items():
            dist = abs(self._positions.get(pos_idx, 0) - self._positions.get(p, 0))
            gi = int(dist * self.n_grid) % self.n_grid
            total += q * self._kernel[gi]
        return total

    def leapfrog_step(self, dt: float = 0.01):
        positions = list(self._positions.keys())
        for p in positions:
            pot = self._potential(p)
            self._velocities[p] = self._velocities.get(p, 0.0) + dt * pot
        for p in positions:
            self._positions[p] += dt * self._velocities[p]

    def get_eviction_order(self) -> list[int]:
        charges = [(p, abs(q)) for p, q in self._charges.items()]
        charges.sort(key=lambda x: x[1])
        return [p for p, _ in charges]


# ═══════════════════════════════════════════════════════════════════════════
# Unified Compressor — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedCompressor:
    """Master orchestrator combining all compression techniques.
    From extreme_kv_cache_v2.py UnifiedKVCompressor — cleanest version.
    Auto-selects optimal combination to achieve target ratio.
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 8,
        target_ratio: float = 100.0,
        max_error: float = 0.001,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.target_ratio = target_ratio
        self.max_error = max_error
        self._kivi = KIVICompressor(dim=dim, k_bits=4, v_bits=4)
        self._gear = GEARCompressor(dim=dim, quant_bits=4, gear_rank=4)
        self._freqkv = FreqKVCompressor(dim=dim, keep_ratio=0.0625)
        self._cross_head = CrossHeadCompressor(dim=dim, n_heads=n_heads)
        self._predictive = PredictiveCompressor(dim=dim)
        self._holographic = HolographicCompressor(dim=dim, n_keep=4, n_heads=n_heads)

    def compress(
        self, keys: np.ndarray, values: np.ndarray, target_ratio: Optional[float] = None
    ) -> dict:
        target = target_ratio or self.target_ratio
        orig = keys.nbytes + values.nbytes
        methods: list[str] = []

        result = self._kivi.compress(keys, values)
        methods.append("KIVI")
        current_ratio = result["ratio"]

        if current_ratio < target * 0.5 and keys.ndim == 2:
            gear_result = self._gear.compress(keys, values)
            methods.append("GEAR")
            current_ratio = max(current_ratio, gear_result["ratio"])
            result = gear_result

        if current_ratio < target * 0.7 and keys.ndim == 2:
            freq_result = self._freqkv.compress(keys, values)
            methods.append("FreqKV")
            current_ratio = max(current_ratio, freq_result["ratio"])
            result = freq_result

        if current_ratio < target * 0.8 and keys.ndim == 3:
            ch_result = self._cross_head.compress(keys, values)
            methods.append("CrossHead")
            current_ratio = max(current_ratio, ch_result["ratio"])
            result = ch_result

        if current_ratio < target * 0.9 and keys.ndim == 2:
            pred_result = self._predictive.compress(keys)
            methods.append("Predictive")
            current_ratio = max(current_ratio, pred_result["ratio"])
            result = pred_result

        if current_ratio < target and keys.ndim == 2:
            holo_result = self._holographic.compress(keys, values)
            methods.append("Holographic")
            current_ratio = max(current_ratio, holo_result["ratio"])
            result = holo_result

        result["methods_used"] = methods
        result["achieved_ratio"] = current_ratio
        return result

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        methods = compressed.get("methods_used", [])
        if "Holographic" in methods and "k_enc" in compressed:
            return self._holographic.decompress(compressed)
        if "FreqKV" in methods and "k_sel" in compressed:
            return self._freqkv.decompress(compressed)
        if "GEAR" in methods and "k_lr" in compressed:
            return self._gear.decompress(compressed)
        if "KIVI" in methods and "k_q" in compressed:
            return self._kivi.decompress(compressed)
        return (
            np.zeros(compressed.get("orig_shape", (1, self.dim)), dtype=np.float32),
            np.zeros(compressed.get("orig_shape", (1, self.dim)), dtype=np.float32),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Quality Validator
# ═══════════════════════════════════════════════════════════════════════════


class KVQualityValidator:
    """Comprehensive retrieval quality metrics.
    From extreme_kv_cache.py KVQualityValidator.
    """

    def validate_pair(
        self,
        k_orig: np.ndarray,
        v_orig: np.ndarray,
        k_comp: np.ndarray,
        v_comp: np.ndarray,
        q: Optional[np.ndarray] = None,
    ) -> dict:
        k_mse = float(np.mean((k_orig - k_comp) ** 2))
        v_mse = float(np.mean((v_orig - v_comp) ** 2))
        k_cos = float(
            cosine_similarity(
                k_orig.ravel().astype(np.float64), k_comp.ravel().astype(np.float64)
            )
        )
        v_cos = float(
            cosine_similarity(
                v_orig.ravel().astype(np.float64), v_comp.ravel().astype(np.float64)
            )
        )
        if q is not None:
            attn_orig = _softmax_core(q @ k_orig.T) @ v_orig
            attn_comp = _softmax_core(q @ k_comp.T) @ v_comp
            attn_error = float(
                np.mean((attn_orig - attn_comp) ** 2) / (np.mean(attn_orig**2) + EPS)
            )
        else:
            attn_error = 0.0
        return {
            "k_mse": k_mse,
            "v_mse": v_mse,
            "k_cosine": k_cos,
            "v_cosine": v_cos,
            "attn_error": attn_error,
        }

    def batch_validate(self, pairs: list) -> dict:
        results = [self.validate_pair(*p) for p in pairs]
        avg = {}
        for key in results[0]:
            avg[key] = sum(r[key] for r in results) / len(results)
        return avg


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED KV CACHE — Main API
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedKVCache:
    """Single clean KV cache interface consolidating all implementations.

    Usage:
        cache = UnifiedKVCache(num_heads=8, head_dim=128, max_seq_len=4096)
        cache.store(layer_idx=0, k=key_tensor, v=value_tensor)
        k, v = cache.retrieve(layer_idx=0, seq_len=100)
        cache.compress(target_ratio=100)
        metrics = cache.get_metrics()
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 4096,
        num_layers: int = 32,
        compression: str = "auto",
        k_bits: int = 4,
        v_bits: int = 4,
        target_ratio: float = 100.0,
        max_error: float = 0.001,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.target_ratio = target_ratio
        self.max_error = max_error

        self._caches: dict[int, dict[int, CacheEntry]] = {
            i: {} for i in range(num_layers)
        }
        self._compressed: dict[int, dict] = {}
        self._eviction = CascadeEviction(max_size=max_seq_len)
        self._vlasov = VlasovEviction()
        self._compressor = UnifiedCompressor(
            dim=head_dim,
            n_heads=num_heads,
            target_ratio=target_ratio,
            max_error=max_error,
        )
        self._validator = KVQualityValidator()
        self._lock = threading.Lock()
        self._metrics = CacheMetrics()
        self._step = 0

        self._compression_mode = compression
        self._per_layer_compressors: dict[int, str] = {}

    def store(
        self,
        layer_idx: int,
        k: np.ndarray,
        v: np.ndarray,
        position: Optional[int] = None,
    ):
        """Store key-value pair for a layer."""
        with self._lock:
            if layer_idx not in self._caches:
                self._caches[layer_idx] = {}
            pos = position if position is not None else len(self._caches[layer_idx])
            entry = CacheEntry(
                key=k.astype(np.float32),
                value=v.astype(np.float32),
                position=pos,
                layer_idx=layer_idx,
                timestamp=time.monotonic(),
            )
            self._caches[layer_idx][pos] = entry
            self._vlasov.register(pos, charge=1.0)
            self._metrics.total_stored += 1
            self._metrics.total_bytes_orig += k.nbytes + v.nbytes
            self._step += 1

    def retrieve(
        self, layer_idx: int, seq_len: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retrieve KV cache for a layer up to seq_len positions."""
        with self._lock:
            cache = self._caches.get(layer_idx, {})
            if not cache:
                return None, None
            positions = sorted(cache.keys())[:seq_len]
            if not positions:
                return None, None
            k_list = [cache[p].key for p in positions]
            v_list = [cache[p].value for p in positions]
            return np.array(k_list, dtype=np.float32), np.array(
                v_list, dtype=np.float32
            )

    def retrieve_single(
        self, layer_idx: int, position: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retrieve a single KV pair."""
        entry = self._caches.get(layer_idx, {}).get(position)
        if entry is None:
            return None, None
        return entry.key, entry.value

    def compress(self, target_ratio: Optional[float] = None):
        """Compress all layers using optimal method selection."""
        target = target_ratio or self.target_ratio
        with self._lock:
            for layer_idx, cache in self._caches.items():
                if not cache:
                    continue
                positions = sorted(cache.keys())
                k_arr = np.array([cache[p].key for p in positions], dtype=np.float32)
                v_arr = np.array([cache[p].value for p in positions], dtype=np.float32)
                if k_arr.ndim == 2:
                    k_arr = k_arr[np.newaxis, :, :]
                    v_arr = v_arr[np.newaxis, :, :]
                result = self._compressor.compress(
                    k_arr.squeeze(0) if k_arr.shape[0] == 1 else k_arr,
                    v_arr.squeeze(0) if v_arr.shape[0] == 1 else v_arr,
                    target_ratio=target,
                )
                self._compressed[layer_idx] = result
                self._metrics.method_used = result.get("methods_used", ["none"])[-1]
                self._metrics.per_layer_ratios[layer_idx] = result.get(
                    "achieved_ratio", 1.0
                )
            if self._compressed:
                ratios = list(self._metrics.per_layer_ratios.values())
                self._metrics.avg_compression_ratio = (
                    sum(ratios) / len(ratios) if ratios else 1.0
                )
                self._metrics.total_bytes_compressed = int(
                    self._metrics.total_bytes_orig
                    / max(self._metrics.avg_compression_ratio, 1.0)
                )

    def decompress_layer(
        self, layer_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Decompress a specific layer's KV cache."""
        compressed = self._compressed.get(layer_idx)
        if compressed is None:
            return self.retrieve(layer_idx, self.max_seq_len)
        return self._compressor.decompress(compressed)

    def evict(self, n_to_evict: int = 1) -> list[int]:
        """Evict least important positions across all layers."""
        with self._lock:
            all_entries: dict[int, CacheEntry] = {}
            for layer_idx, cache in self._caches.items():
                for pos, entry in cache.items():
                    if pos not in all_entries:
                        all_entries[pos] = entry
            positions = self._eviction.evict_positions(all_entries, n_to_evict)
            for pos in positions:
                for layer_idx in self._caches:
                    self._caches[layer_idx].pop(pos, None)
            self._metrics.eviction_count += n_to_evict
            return positions

    def query(
        self, layer_idx: int, query_vector: np.ndarray, top_k: int = 10
    ) -> list[Tuple[int, float]]:
        """Query cache with attention-like scoring."""
        cache = self._caches.get(layer_idx, {})
        if not cache:
            return []
        scores = []
        q = query_vector.astype(np.float64).ravel()
        for pos, entry in cache.items():
            k = entry.key.astype(np.float64).ravel()
            score = float(np.dot(q, k) / (np.linalg.norm(q) * np.linalg.norm(k) + EPS))
            scores.append((pos, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def update_attention(self, layer_idx: int, position: int, score: float):
        """Update attention score for a position."""
        self._eviction.record_attention(position, score)

    def get_metrics(self) -> CacheMetrics:
        """Return current cache metrics."""
        total_entries = sum(len(c) for c in self._caches.values())
        self._metrics.total_stored = total_entries
        hit_positions = len(self._eviction._attention_scores)
        self._metrics.hit_rate = hit_positions / max(total_entries, 1)
        return self._metrics

    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._caches = {i: {} for i in range(self.num_layers)}
            self._compressed.clear()
            self._metrics = CacheMetrics()

    def layer_count(self) -> int:
        """Number of layers with stored entries."""
        return sum(1 for c in self._caches.values() if c)

    def position_count(self) -> int:
        """Total unique positions across all layers."""
        all_pos = set()
        for cache in self._caches.values():
            all_pos.update(cache.keys())
        return len(all_pos)

    def cache_summary(self) -> dict:
        """Return comprehensive summary."""
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "total_entries": self.metrics.total_stored
            if hasattr(self, "_metrics")
            else 0,
            "position_count": self.position_count(),
            "compression_mode": self._compression_mode,
            "avg_compression_ratio": self._metrics.avg_compression_ratio,
            "method_used": self._metrics.method_used,
            "per_layer_ratios": dict(self._metrics.per_layer_ratios),
            "eviction_count": self._metrics.eviction_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════════


def create_unified_kv_cache(
    num_heads: int = 8,
    head_dim: int = 128,
    max_seq_len: int = 4096,
    num_layers: int = 32,
    compression: str = "auto",
    config: Optional[dict] = None,
) -> UnifiedKVCache:
    """Factory for creating UnifiedKVCache with optional config dict."""
    cfg = config or {}
    return UnifiedKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        compression=cfg.get("compression", compression),
        k_bits=cfg.get("k_bits", 4),
        v_bits=cfg.get("v_bits", 4),
        target_ratio=cfg.get("target_ratio", 100.0),
        max_error=cfg.get("max_error", 0.001),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility Aliases
# ═══════════════════════════════════════════════════════════════════════════

KVCacheManager = UnifiedKVCache
create_kv_cache = create_unified_kv_cache

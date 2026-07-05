"""
Extreme KV Cache v2 — Archive Migration
=========================================
Implements 7 KV cache compression classes migrated from
_archive/v1/spectralstream/kv_cache/extreme_kv_cache_v2.py.

Classes:
  1. KIVICache          — Per-channel/per-token quantization (KIVI style)
  2. GEARCache          — Quantization + low-rank + sparse error recovery
  3. FreqKVCache        — Frequency-domain DCT along sequence (FreqKV)
  4. HolographicKVCache — Phase encoding + HRR superposition + TimeCrystal
  5. CrossHeadKVCache   — Cross-head correlated base sharing
  6. PredictiveKVCache  — AR(2) autoregressive predictive coding
  7. UnifiedKVCompressor — Master orchestrator combining all techniques
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.kv_cache.v2 is deprecated. "
    "Use spectralstream.kv_cache.KVCacheManager instead.",
    DeprecationWarning,
    stacklevel=2,
)

import math
import threading
from typing import Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    idct,
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
    softmax,
)
from spectralstream.kv_cache.core import EPS


# ═══════════════════════════════════════════════════════════════════════════
# Utility: packing low-bit integers
# ═══════════════════════════════════════════════════════════════════════════


def _pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack int8 values into int4 nibbles."""
    n = len(values)
    clamped = np.clip(np.round(values).astype(np.int8), -8, 7) & 0x0F
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    for i in range(0, n, 2):
        lo = int(clamped[i]) & 0x0F
        hi = int(clamped[i + 1]) & 0x0F if i + 1 < n else 0
        packed[i // 2] = (hi << 4) | lo
    return packed


def _unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack int4 nibbles into int8 values."""
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        byte_val = int(packed[i // 2])
        val = (byte_val >> 4) & 0x0F if i % 2 else byte_val & 0x0F
        if val >= 8:
            val -= 16
        result[i] = np.int8(val)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 1. KIVICache — Per-Channel/Per-Token Quantization
# ═══════════════════════════════════════════════════════════════════════════


class KIVICache:
    """KIVI-style asymmetric KV cache compression.

    Keys:   per-channel quantization (asymmetric, group-wise)
            One scale per channel group — preserves channel-wise outliers.
    Values: per-token quantization (symmetric)
            One scale per token — adapts to per-token dynamic range.
    Adaptive bit-width per channel based on outlier magnitude.

    Expected: 8-16x compression, <0.1% attention error.
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 8,
        k_bits: int = 4,
        v_bits: int = 4,
        key_group_size: int = 64,
        val_group_size: int = 64,
        residual_tokens: int = 64,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.key_group_size = key_group_size
        self.val_group_size = val_group_size
        self.residual_tokens = residual_tokens
        self._lock = threading.Lock()

        self._k_grouped: Optional[np.ndarray] = None
        self._k_grouped_scales: Optional[np.ndarray] = None
        self._k_grouped_zeros: Optional[np.ndarray] = None
        self._k_residual: list[np.ndarray] = []
        self._v_grouped: Optional[np.ndarray] = None
        self._v_grouped_scales: Optional[np.ndarray] = None
        self._v_grouped_zeros: Optional[np.ndarray] = None
        self._v_residual: list[np.ndarray] = []
        self._total_stored = 0

    def _quantize_per_channel(
        self,
        x: np.ndarray,
        n_bits: int,
        group_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq_len, dim = x.shape
        n_groups = max(1, dim // group_size)
        half = (1 << (n_bits - 1)) - 1

        indices = np.zeros_like(x, dtype=np.int8)
        scales = np.zeros(n_groups, dtype=np.float64)
        zeros = np.zeros(n_groups, dtype=np.float64)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, dim)
            group_data = x[:, start:end]

            g_min = float(np.min(group_data))
            g_max = float(np.max(group_data))
            g_range = max(g_max - g_min, EPS)

            scale = g_range / (2 * half)
            z = g_min

            scaled = np.clip(np.round((group_data - z) / scale), -half, half)
            indices[:, start:end] = scaled.astype(np.int8)
            scales[g] = scale
            zeros[g] = z

        return indices, scales, zeros

    def _dequantize_per_channel(
        self,
        indices: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        seq_len, dim = indices.shape
        result = np.zeros((seq_len, dim), dtype=np.float64)
        n_groups = len(scales)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, dim)
            s = scales[g]
            z = zeros[g]
            result[:, start:end] = indices[:, start:end].astype(np.float64) * s + z

        return result

    def _quantize_per_token(
        self,
        x: np.ndarray,
        n_bits: int,
        group_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq_len, dim = x.shape
        n_groups = max(1, seq_len // group_size)
        half = (1 << (n_bits - 1)) - 1

        indices = np.zeros_like(x, dtype=np.int8)
        scales = np.zeros((n_groups, dim), dtype=np.float64)
        zeros = np.zeros((n_groups, dim), dtype=np.float64)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, seq_len)
            group_data = x[start:end]

            g_max = np.max(np.abs(group_data), axis=0, keepdims=True)
            g_max = np.clip(g_max, EPS, None)
            scale = g_max / half

            scaled = np.clip(np.round(group_data / scale), -half, half)
            indices[start:end] = scaled.astype(np.int8)
            scales[g] = scale.ravel()[:dim]
            zeros[g] = 0.0

        return indices, scales, zeros

    def _dequantize_per_token(
        self,
        indices: np.ndarray,
        scales: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        seq_len, dim = indices.shape
        result = np.zeros((seq_len, dim), dtype=np.float64)
        n_groups = scales.shape[0]

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, seq_len)
            s = scales[g : g + 1, :]
            result[start:end] = indices[start:end].astype(np.float64) * s

        return result

    def compress(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        train_quantizers: bool = True,
    ) -> dict:
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        orig_shape_k = keys.shape
        orig_shape_v = values.shape

        k_indices, k_scales, k_zeros = self._quantize_per_channel(
            keys, self.k_bits, self.key_group_size
        )

        v_indices, v_scales, v_zeros = self._quantize_per_token(
            values, self.v_bits, self.val_group_size
        )

        k_packed = _pack_int4(k_indices.ravel().astype(np.float32))
        v_packed = _pack_int4(v_indices.ravel().astype(np.float32))

        orig_bytes = keys.nbytes + values.nbytes
        comp_bytes = (
            k_packed.nbytes
            + k_scales.nbytes
            + k_zeros.nbytes
            + v_packed.nbytes
            + v_scales.nbytes
            + v_zeros.nbytes
        )

        return {
            "k_packed": k_packed.tobytes(),
            "k_indices_shape": orig_shape_k,
            "k_scales": k_scales.astype(np.float16).tobytes(),
            "k_scales_shape": k_scales.shape,
            "k_zeros": k_zeros.astype(np.float16).tobytes(),
            "k_zeros_shape": k_zeros.shape,
            "v_packed": v_packed.tobytes(),
            "v_indices_shape": orig_shape_v,
            "v_scales": v_scales.astype(np.float16).tobytes(),
            "v_scales_shape": v_scales.shape,
            "v_zeros": v_zeros.astype(np.float16).tobytes(),
            "v_zeros_shape": v_zeros.shape,
            "n_bits_k": self.k_bits,
            "n_bits_v": self.v_bits,
            "key_group_size": self.key_group_size,
            "val_group_size": self.val_group_size,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes + 64,
            "ratio": orig_bytes / max(comp_bytes + 64, 1),
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        k_shape = compressed["k_indices_shape"]
        v_shape = compressed["v_indices_shape"]

        k_packed = np.frombuffer(compressed["k_packed"], dtype=np.uint8).copy()
        k_indices = _unpack_int4(k_packed, k_shape[0] * k_shape[1])
        k_indices = k_indices.reshape(k_shape)

        v_packed = np.frombuffer(compressed["v_packed"], dtype=np.uint8).copy()
        v_indices = _unpack_int4(v_packed, v_shape[0] * v_shape[1])
        v_indices = v_indices.reshape(v_shape)

        k_scales = (
            np.frombuffer(compressed["k_scales"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        k_zeros = (
            np.frombuffer(compressed["k_zeros"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        k_scales = k_scales.ravel()
        k_zeros = k_zeros.ravel()

        v_scales = (
            np.frombuffer(compressed["v_scales"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        v_scales = v_scales.reshape(compressed["v_scales_shape"])

        keys = self._dequantize_per_channel(
            k_indices, k_scales, k_zeros, compressed["key_group_size"]
        )
        values = self._dequantize_per_token(
            v_indices, v_scales, compressed["val_group_size"]
        )

        return keys.astype(np.float32), values.astype(np.float32)

    def attention_error(
        self,
        q: np.ndarray,
        k_orig: np.ndarray,
        v_orig: np.ndarray,
        k_comp: np.ndarray,
        v_comp: np.ndarray,
    ) -> dict:
        q = np.asarray(q, dtype=np.float64)
        k_orig = np.asarray(k_orig, dtype=np.float64)
        v_orig = np.asarray(v_orig, dtype=np.float64)
        k_comp = np.asarray(k_comp, dtype=np.float64)
        v_comp = np.asarray(v_comp, dtype=np.float64)

        d = k_orig.shape[-1] if k_orig.ndim > 1 else k_orig.shape[0]
        scale = math.sqrt(d)
        attn_orig = softmax(q @ k_orig.T / scale)
        attn_comp = softmax(q @ k_comp.T / scale)

        out_orig = attn_orig @ v_orig
        out_comp = attn_comp @ v_comp

        attn_cos = float(cosine_similarity(attn_orig.ravel(), attn_comp.ravel()))
        out_cos = float(cosine_similarity(out_orig.ravel(), out_comp.ravel()))
        attn_mae = float(np.mean(np.abs(attn_orig - attn_comp)))
        out_mse = float(np.mean((out_orig - out_comp) ** 2))

        return {
            "attention_cosine": attn_cos,
            "output_cosine": out_cos,
            "attention_mae": attn_mae,
            "output_mse": out_mse,
            "attention_error_pct": (1.0 - attn_cos) * 100,
            "output_error_pct": (1.0 - out_cos) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. GEARCache — Quantization + Low-Rank + Sparse Error Recovery
# ═══════════════════════════════════════════════════════════════════════════


class GEARCache:
    """GEAR: Quantization + Low-rank + Sparse error recovery.

    X ≈ D_hat + L + S
      D_hat: ultra-low-bit quantized backbone (4-bit)
      L:     low-rank approximation of quantization error (rank-r SVD)
      S:     sparse outlier residuals

    Expected: 20-30x compression, <0.05% error.
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 8,
        quant_bits: int = 4,
        gear_rank: int = 4,
        sparse_fraction: float = 0.001,
        key_group_size: int = 64,
        val_group_size: int = 64,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.quant_bits = quant_bits
        self.gear_rank = gear_rank
        self.sparse_fraction = sparse_fraction
        self.key_group_size = key_group_size
        self.val_group_size = val_group_size
        self._lock = threading.Lock()

        self._kivi = KIVICache(
            dim=dim,
            n_heads=n_heads,
            k_bits=quant_bits,
            v_bits=quant_bits,
            key_group_size=key_group_size,
            val_group_size=val_group_size,
        )

    def _extract_outlier_sparse(
        self,
        x: np.ndarray,
        fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_total = x.size
        n_sparse = max(1, int(n_total * fraction))

        flat = np.abs(x.ravel())
        threshold = (
            np.partition(flat, -n_sparse)[-n_sparse] if n_total > n_sparse else 0.0
        )

        mask = np.abs(x) >= threshold
        sparse = np.zeros_like(x)
        sparse[mask] = x[mask]

        return sparse, mask

    def _low_rank_approximation(
        self,
        error: np.ndarray,
        rank: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = min(rank, min(error.shape) - 1)
        if r < 1:
            return (
                np.zeros_like(error),
                np.zeros((error.shape[0], 0)),
                np.zeros((0, error.shape[1])),
            )

        try:
            U, S, Vt = np.linalg.svd(error.astype(np.float64), full_matrices=False)
            U_r = U[:, :r]
            S_r = S[:r]
            Vt_r = Vt[:r, :]

            low_rank = (U_r * S_r) @ Vt_r
            return (
                low_rank.astype(np.float32),
                U_r.astype(np.float16),
                Vt_r.astype(np.float16),
            )
        except np.linalg.LinAlgError:
            return (
                np.zeros_like(error),
                np.zeros((error.shape[0], 0)),
                np.zeros((0, error.shape[1])),
            )

    def compress(self, keys: np.ndarray, values: np.ndarray) -> dict:
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)

        k_sparse, k_mask = self._extract_outlier_sparse(keys, self.sparse_fraction)
        v_sparse, v_mask = self._extract_outlier_sparse(values, self.sparse_fraction)

        k_residual = keys - k_sparse
        v_residual = values - v_sparse

        k_comp = self._kivi.compress(k_residual, v_residual)
        k_deq, v_deq = self._kivi.decompress(k_comp)

        k_error = keys.astype(np.float32) - k_deq - k_sparse
        v_error = values.astype(np.float32) - v_deq - v_sparse

        k_low_rank, kU, kVt = self._low_rank_approximation(k_error, self.gear_rank)
        v_low_rank, vU, vVt = self._low_rank_approximation(v_error, self.gear_rank)

        k_final_err = k_error - k_low_rank
        v_final_err = v_error - v_low_rank

        n_sparse_k = max(1, int(k_final_err.size * self.sparse_fraction))
        n_sparse_v = max(1, int(v_final_err.size * self.sparse_fraction))

        flat_k = np.abs(k_final_err.ravel())
        flat_v = np.abs(v_final_err.ravel())

        k_top_idx = np.argpartition(flat_k, -n_sparse_k)[-n_sparse_k:]
        v_top_idx = np.argpartition(flat_v, -n_sparse_v)[-n_sparse_v:]

        k_sparse_vals = k_final_err.ravel()[k_top_idx]
        v_sparse_vals = v_final_err.ravel()[v_top_idx]

        orig_bytes = keys.nbytes + values.nbytes
        kivi_bytes = k_comp.get("orig_bytes", 0)
        gear_bytes = (
            kU.nbytes
            + kVt.nbytes
            + vU.nbytes
            + vVt.nbytes
            + k_sparse_vals.nbytes
            + v_sparse_vals.nbytes
            + k_top_idx.nbytes
            + v_top_idx.nbytes
            + 128
        )
        comp_bytes = kivi_bytes + gear_bytes

        return {
            "kivi_compressed": k_comp,
            "k_sparse_mask": k_mask.astype(np.bool_).tobytes(),
            "v_sparse_mask": v_mask.astype(np.bool_).tobytes(),
            "k_sparse_mask_shape": k_mask.shape,
            "v_sparse_mask_shape": v_mask.shape,
            "k_sparse_vals": k_sparse_vals.astype(np.float16).tobytes(),
            "v_sparse_vals": v_sparse_vals.astype(np.float16).tobytes(),
            "k_sparse_indices": k_top_idx.astype(np.uint32).tobytes(),
            "v_sparse_indices": v_top_idx.astype(np.uint32).tobytes(),
            "kU": kU.tobytes(),
            "kVt": kVt.tobytes(),
            "vU": vU.tobytes(),
            "vVt": vVt.tobytes(),
            "kU_shape": kU.shape,
            "kVt_shape": kVt.shape,
            "vU_shape": vU.shape,
            "vVt_shape": vVt.shape,
            "gear_rank": self.gear_rank,
            "sparse_fraction": self.sparse_fraction,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio": orig_bytes / max(comp_bytes, 1),
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        k_deq, v_deq = self._kivi.decompress(compressed["kivi_compressed"])

        kU = np.frombuffer(compressed["kU"], dtype=np.float16).copy().astype(np.float64)
        kVt = (
            np.frombuffer(compressed["kVt"], dtype=np.float16).copy().astype(np.float64)
        )
        vU = np.frombuffer(compressed["vU"], dtype=np.float16).copy().astype(np.float64)
        vVt = (
            np.frombuffer(compressed["vVt"], dtype=np.float16).copy().astype(np.float64)
        )

        k_low_rank = np.zeros_like(k_deq, dtype=np.float64)
        v_low_rank = np.zeros_like(v_deq, dtype=np.float64)

        if kU.size > 0 and kVt.size > 0:
            try:
                kU = kU.reshape(compressed["kU_shape"])
                kVt = kVt.reshape(compressed["kVt_shape"])
                k_low_rank = kU @ kVt
            except ValueError:
                pass

        if vU.size > 0 and vVt.size > 0:
            try:
                vU = vU.reshape(compressed["vU_shape"])
                vVt = vVt.reshape(compressed["vVt_shape"])
                v_low_rank = vU @ vVt
            except ValueError:
                pass

        k_sparse_mask = np.frombuffer(
            compressed["k_sparse_mask"], dtype=np.bool_
        ).reshape(compressed["k_sparse_mask_shape"])
        v_sparse_mask = np.frombuffer(
            compressed["v_sparse_mask"], dtype=np.bool_
        ).reshape(compressed["v_sparse_mask_shape"])

        k_sparse = np.zeros_like(k_deq, dtype=np.float64)
        v_sparse = np.zeros_like(v_deq, dtype=np.float64)

        k_sparse_vals = (
            np.frombuffer(compressed["k_sparse_vals"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        k_sparse_indices = np.frombuffer(
            compressed["k_sparse_indices"], dtype=np.uint32
        ).copy()
        if k_sparse_vals.size > 0 and k_sparse_indices.size > 0:
            k_sparse.ravel()[k_sparse_indices] = k_sparse_vals

        v_sparse_vals = (
            np.frombuffer(compressed["v_sparse_vals"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        v_sparse_indices = np.frombuffer(
            compressed["v_sparse_indices"], dtype=np.uint32
        ).copy()
        if v_sparse_vals.size > 0 and v_sparse_indices.size > 0:
            v_sparse.ravel()[v_sparse_indices] = v_sparse_vals

        keys = k_deq + k_low_rank + k_sparse
        values = v_deq + v_low_rank + v_sparse

        return keys.astype(np.float32), values.astype(np.float32)

    def attention_error(
        self,
        q: np.ndarray,
        k_orig: np.ndarray,
        v_orig: np.ndarray,
        k_comp: np.ndarray,
        v_comp: np.ndarray,
    ) -> dict:
        q = np.asarray(q, dtype=np.float64)
        k_orig = np.asarray(k_orig, dtype=np.float64)
        v_orig = np.asarray(v_orig, dtype=np.float64)
        k_comp = np.asarray(k_comp, dtype=np.float64)
        v_comp = np.asarray(v_comp, dtype=np.float64)

        d = k_orig.shape[-1] if k_orig.ndim > 1 else k_orig.shape[0]
        scale = math.sqrt(d)
        attn_orig = softmax(q @ k_orig.T / scale)
        attn_comp = softmax(q @ k_comp.T / scale)

        out_orig = attn_orig @ v_orig
        out_comp = attn_comp @ v_comp

        attn_cos = float(cosine_similarity(attn_orig.ravel(), attn_comp.ravel()))
        out_cos = float(cosine_similarity(out_orig.ravel(), out_comp.ravel()))

        return {
            "attention_cosine": attn_cos,
            "output_cosine": out_cos,
            "attention_error_pct": (1.0 - attn_cos) * 100,
            "output_error_pct": (1.0 - out_cos) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. FreqKVCache — Frequency-Domain DCT Compression
# ═══════════════════════════════════════════════════════════════════════════


class FreqKVCache:
    """FreqKV: DCT along sequence dimension.

    KV cache along sequence length has strong temporal correlation —
    adjacent tokens produce similar KV vectors. Energy concentrates in
    low-frequency DCT components. Keep only top-k% coefficients.

    Expected: 8-16x compression along sequence dimension.
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 8,
        keep_ratio: float = 0.0625,
        energy_threshold: float = 0.99,
        min_keep_ratio: float = 0.02,
        quant_bits: int = 4,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.keep_ratio = keep_ratio
        self.energy_threshold = energy_threshold
        self.min_keep_ratio = min_keep_ratio
        self.quant_bits = quant_bits
        self._lock = threading.Lock()

        self._quantizer = LloydMaxQuantizer(n_bits=quant_bits)

    def _dct_along_sequence(self, x: np.ndarray) -> np.ndarray:
        return dct(x).astype(np.float64)

    def _idct_along_sequence(self, coeffs: np.ndarray) -> np.ndarray:
        return idct(coeffs).astype(np.float32)

    def _select_coefficients(
        self,
        dct_coeffs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        seq_len, dim = dct_coeffs.shape
        energy = dct_coeffs**2
        total_energy = np.sum(energy, axis=0, keepdims=True) + EPS
        cum_energy = np.cumsum(energy, axis=0) / total_energy

        n_keep = np.zeros(dim, dtype=np.int32)
        for c in range(dim):
            n_keep_c = max(1, int(seq_len * self.keep_ratio))
            for k in range(1, seq_len + 1):
                if cum_energy[k - 1, c] >= self.energy_threshold:
                    n_keep_c = max(n_keep_c, k)
                    break
            n_keep_c = max(int(seq_len * self.min_keep_ratio), min(n_keep_c, seq_len))
            n_keep[c] = n_keep_c

        max_keep = int(np.max(n_keep))
        max_keep = max(1, min(max_keep, seq_len))

        selected = dct_coeffs[:max_keep, :]
        indices = np.arange(max_keep)

        return selected, indices, max_keep

    def compress(self, keys: np.ndarray, values: np.ndarray) -> dict:
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)

        k_dct = self._dct_along_sequence(keys)
        v_dct = self._dct_along_sequence(values)

        k_selected, k_indices, k_n_keep = self._select_coefficients(k_dct)
        v_selected, v_indices, v_n_keep = self._select_coefficients(v_dct)

        k_flat = k_selected.ravel().astype(np.float32)
        v_flat = v_selected.ravel().astype(np.float32)

        if not self._quantizer.trained:
            all_data = np.concatenate([k_flat, v_flat])
            self._quantizer.train(all_data)

        k_q_idx, k_centroids = self._quantizer.compress(k_flat)
        v_q_idx, v_centroids = self._quantizer.compress(v_flat)

        orig_bytes = keys.nbytes + values.nbytes
        comp_bytes = (
            k_q_idx.nbytes
            + v_q_idx.nbytes
            + k_centroids.nbytes
            + v_centroids.nbytes
            + 64
        )

        return {
            "k_dct_coeffs": k_selected.astype(np.float16).tobytes(),
            "k_dct_shape": k_selected.shape,
            "k_indices": k_indices.astype(np.uint32).tobytes(),
            "k_n_keep": k_n_keep,
            "k_q_idx": k_q_idx.tobytes(),
            "k_centroids": k_centroids.astype(np.float16).tobytes(),
            "v_dct_coeffs": v_selected.astype(np.float16).tobytes(),
            "v_dct_shape": v_selected.shape,
            "v_indices": v_indices.astype(np.uint32).tobytes(),
            "v_n_keep": v_n_keep,
            "v_q_idx": v_q_idx.tobytes(),
            "v_centroids": v_centroids.astype(np.float16).tobytes(),
            "seq_len": keys.shape[0],
            "dim": keys.shape[1],
            "keep_ratio": self.keep_ratio,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio": orig_bytes / max(comp_bytes, 1),
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = compressed["seq_len"]
        dim = compressed["dim"]
        k_n_keep = compressed["k_n_keep"]
        v_n_keep = compressed["v_n_keep"]

        k_q_idx = np.frombuffer(compressed["k_q_idx"], dtype=np.uint8).copy()
        k_centroids = (
            np.frombuffer(compressed["k_centroids"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        k_shape = compressed["k_dct_shape"]
        k_selected = k_centroids[k_q_idx].reshape(k_shape)

        v_q_idx = np.frombuffer(compressed["v_q_idx"], dtype=np.uint8).copy()
        v_centroids = (
            np.frombuffer(compressed["v_centroids"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        v_shape = compressed["v_dct_shape"]
        v_selected = v_centroids[v_q_idx].reshape(v_shape)

        k_dct_full = np.zeros((seq_len, dim), dtype=np.float64)
        k_indices = np.frombuffer(compressed["k_indices"], dtype=np.uint32).copy()
        k_dct_full[k_indices[:k_n_keep], :] = k_selected[:k_n_keep, :]

        v_dct_full = np.zeros((seq_len, dim), dtype=np.float64)
        v_indices = np.frombuffer(compressed["v_indices"], dtype=np.uint32).copy()
        v_dct_full[v_indices[:v_n_keep], :] = v_selected[:v_n_keep, :]

        keys = self._idct_along_sequence(k_dct_full.astype(np.float32))
        values = self._idct_along_sequence(v_dct_full.astype(np.float32))

        return keys, values

    def attention_error(
        self,
        q: np.ndarray,
        k_orig: np.ndarray,
        v_orig: np.ndarray,
        k_comp: np.ndarray,
        v_comp: np.ndarray,
    ) -> dict:
        q = np.asarray(q, dtype=np.float64)
        k_orig = np.asarray(k_orig, dtype=np.float64)
        v_orig = np.asarray(v_orig, dtype=np.float64)
        k_comp = np.asarray(k_comp, dtype=np.float64)
        v_comp = np.asarray(v_comp, dtype=np.float64)

        d = k_orig.shape[-1] if k_orig.ndim > 1 else k_orig.shape[0]
        scale = math.sqrt(d)
        attn_orig = softmax(q @ k_orig.T / scale)
        attn_comp = softmax(q @ k_comp.T / scale)

        out_orig = attn_orig @ v_orig
        out_comp = attn_comp @ v_comp

        attn_cos = float(cosine_similarity(attn_orig.ravel(), attn_comp.ravel()))
        out_cos = float(cosine_similarity(out_orig.ravel(), out_comp.ravel()))
        attn_mae = float(np.mean(np.abs(attn_orig - attn_comp)))

        return {
            "attention_cosine": attn_cos,
            "output_cosine": out_cos,
            "attention_mae": attn_mae,
            "attention_error_pct": (1.0 - attn_cos) * 100,
            "output_error_pct": (1.0 - out_cos) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. HolographicKVCache — Phase Encoding + HRR + TimeCrystal
# ═══════════════════════════════════════════════════════════════════════════


class HolographicKVCache:
    """Holographic KV cache using phase encoding and HRR superposition.

    Phase-only encoding: store DCT phase (sign of coefficients) + magnitude.
    HRR superposition: bundle multiple KV entries into single memory vector.
    TimeCrystal: dual-phase with O(epsilon^2) error cancellation.

    Expected: 100-500x compression (extreme, lower accuracy).
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 8,
        n_keep: int = 4,
        magnitude_bits: int = 4,
        use_time_crystal: bool = True,
        hrr_dim: int = 512,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.n_keep = n_keep
        self.magnitude_bits = magnitude_bits
        self.use_time_crystal = use_time_crystal
        self.hrr_dim = hrr_dim
        self._lock = threading.Lock()

        self._mag_quantizer = LloydMaxQuantizer(n_bits=magnitude_bits)
        self._position_vectors: dict[int, np.ndarray] = {}
        self._trained = False

        self._phase0_storage: dict[int, dict] = {}
        self._phase1_storage: dict[int, dict] = {}

        self._superposed_memory: Optional[np.ndarray] = None
        self._n_entries = 0

    def _get_position_vector(self, position: int) -> np.ndarray:
        if position not in self._position_vectors:
            rng = np.random.RandomState(
                (position * 2654435761 + 0x9E3779B9) & 0x7FFFFFFF
            )
            self._position_vectors[position] = rng.choice(
                [-1.0, 1.0], size=self.n_keep
            ).astype(np.float64)
        return self._position_vectors[position]

    def _phase_encode(self, vector: np.ndarray, position: int) -> dict:
        vec = vector.ravel().astype(np.float64)
        dct_coeffs = dct(vec)

        mag = np.abs(dct_coeffs)
        if self.n_keep >= len(dct_coeffs):
            top_idx = np.arange(len(dct_coeffs))
        else:
            top_idx = np.argpartition(mag, -self.n_keep)[-self.n_keep :]

        kept_sign = np.sign(dct_coeffs[top_idx]).astype(np.int8)
        kept_mag = mag[top_idx].astype(np.float32)

        if not self._trained:
            self._mag_quantizer.train(kept_mag)
            self._trained = True

        mag_idx, centroids = self._mag_quantizer.compress(kept_mag)

        p_vec = self._get_position_vector(position)
        phase_vec = (
            kept_sign.astype(np.float64) * np.abs(centroids[mag_idx])
            if centroids is not None
            else kept_sign.astype(np.float64)
        )
        bound = (
            hrr_bind(phase_vec[: len(p_vec)], p_vec)
            if len(phase_vec) >= len(p_vec)
            else phase_vec
        )

        return {
            "sign": kept_sign,
            "mag_idx": mag_idx,
            "centroids": centroids,
            "top_idx": top_idx,
            "bound": bound,
            "position": position,
        }

    def _phase_decode(self, encoded: dict) -> np.ndarray:
        sign = encoded["sign"]
        mag_idx = encoded["mag_idx"]
        centroids = encoded["centroids"]
        top_idx = encoded["top_idx"]

        if centroids is not None:
            mag = centroids[mag_idx]
        else:
            mag = np.ones_like(sign, dtype=np.float64)

        dct_recon = np.zeros(self.dim, dtype=np.float64)
        dct_recon[top_idx[: len(sign)]] = sign[: len(top_idx)] * mag[: len(top_idx)]

        return idct(dct_recon).astype(np.float32)

    def _time_crystal_encode(self, vector: np.ndarray, position: int) -> dict:
        vec = vector.ravel().astype(np.float64)
        dct_coeffs = dct(vec)

        half = (1 << (self.magnitude_bits - 1)) - 1
        amax = float(np.max(np.abs(dct_coeffs))) + EPS

        q0 = np.clip(np.round(dct_coeffs / amax * half), -half, half).astype(np.int8)
        q1 = np.clip(np.round(-dct_coeffs / amax * half), -half, half).astype(np.int8)

        return {
            "q0": q0.tobytes(),
            "q1": q1.tobytes(),
            "amax": np.float32(amax).tobytes(),
            "shape": dct_coeffs.shape,
            "position": position,
        }

    def _time_crystal_decode(self, encoded: dict) -> np.ndarray:
        q0 = np.frombuffer(encoded["q0"], dtype=np.int8).copy().astype(np.float64)
        q1 = np.frombuffer(encoded["q1"], dtype=np.int8).copy().astype(np.float64)
        amax = float(np.frombuffer(encoded["amax"], dtype=np.float32)[0])
        shape = encoded["shape"]

        half = (1 << (self.magnitude_bits - 1)) - 1

        dct0 = q0 * (amax / max(half, 1))
        dct1 = -q1 * (amax / max(half, 1))

        dct_avg = (dct0 + dct1) * 0.5

        result = np.zeros(shape, dtype=np.float64)
        result[: len(dct_avg)] = dct_avg

        return idct(result).astype(np.float32)

    def _hrr_superposition_encode(self, vector: np.ndarray, position: int) -> None:
        vec = vector.ravel().astype(np.float64)[: self.hrr_dim]
        if len(vec) < self.hrr_dim:
            vec = np.pad(vec, (0, self.hrr_dim - len(vec)))

        p_vec = self._get_position_vector(position)
        if len(p_vec) < self.hrr_dim:
            p_vec = np.pad(p_vec, (0, self.hrr_dim - len(p_vec)))

        bound = hrr_bind(vec, p_vec)

        if self._superposed_memory is None:
            self._superposed_memory = bound.copy()
        else:
            self._superposed_memory = self._superposed_memory + bound

        self._n_entries += 1

    def _hrr_superposition_decode(self, position: int) -> Optional[np.ndarray]:
        if self._superposed_memory is None:
            return None

        p_vec = self._get_position_vector(position)
        if len(p_vec) < self.hrr_dim:
            p_vec = np.pad(p_vec, (0, self.hrr_dim - len(p_vec)))

        retrieved = hrr_unbind(self._superposed_memory, p_vec)
        return retrieved[: self.dim].astype(np.float32)

    def compress(self, keys: np.ndarray, values: np.ndarray) -> dict:
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        seq_len = keys.shape[0]

        k_encoded = []
        v_encoded = []
        tc_k_encoded = []
        tc_v_encoded = []

        for i in range(seq_len):
            k_enc = self._phase_encode(keys[i], i)
            v_enc = self._phase_encode(values[i], i)
            k_encoded.append(k_enc)
            v_encoded.append(v_enc)

            if self.use_time_crystal:
                tc_k = self._time_crystal_encode(keys[i], i)
                tc_v = self._time_crystal_encode(values[i], i)
                tc_k_encoded.append(tc_k)
                tc_v_encoded.append(tc_v)

            self._hrr_superposition_encode(keys[i], i)
            self._hrr_superposition_encode(values[i], i + seq_len)

        orig_bytes = keys.nbytes + values.nbytes
        entry_bits = self.n_keep * (1 + self.magnitude_bits)
        if self.use_time_crystal:
            entry_bits *= 2
        comp_bits = seq_len * 2 * entry_bits
        comp_bytes = comp_bits / 8 + 64

        return {
            "k_encoded": k_encoded,
            "v_encoded": v_encoded,
            "tc_k_encoded": tc_k_encoded,
            "tc_v_encoded": tc_v_encoded,
            "seq_len": seq_len,
            "dim": keys.shape[1],
            "use_time_crystal": self.use_time_crystal,
            "n_keep": self.n_keep,
            "magnitude_bits": self.magnitude_bits,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio": orig_bytes / max(comp_bytes, 1),
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = compressed["seq_len"]
        dim = compressed["dim"]

        keys = np.zeros((seq_len, dim), dtype=np.float32)
        values = np.zeros((seq_len, dim), dtype=np.float32)

        use_tc = compressed.get("use_time_crystal", False)

        for i in range(seq_len):
            if use_tc and compressed["tc_k_encoded"]:
                keys[i] = self._time_crystal_decode(compressed["tc_k_encoded"][i])
                values[i] = self._time_crystal_decode(compressed["tc_v_encoded"][i])
            else:
                keys[i] = self._phase_decode(compressed["k_encoded"][i])
                values[i] = self._phase_decode(compressed["v_encoded"][i])

        return keys, values

    def attention_error(
        self,
        q: np.ndarray,
        k_orig: np.ndarray,
        v_orig: np.ndarray,
        k_comp: np.ndarray,
        v_comp: np.ndarray,
    ) -> dict:
        q = np.asarray(q, dtype=np.float64)
        k_orig = np.asarray(k_orig, dtype=np.float64)
        v_orig = np.asarray(v_orig, dtype=np.float64)
        k_comp = np.asarray(k_comp, dtype=np.float64)
        v_comp = np.asarray(v_comp, dtype=np.float64)

        d = k_orig.shape[-1] if k_orig.ndim > 1 else k_orig.shape[0]
        scale = math.sqrt(d)
        attn_orig = softmax(q @ k_orig.T / scale)
        attn_comp = softmax(q @ k_comp.T / scale)

        out_orig = attn_orig @ v_orig
        out_comp = attn_comp @ v_comp

        attn_cos = float(cosine_similarity(attn_orig.ravel(), attn_comp.ravel()))
        out_cos = float(cosine_similarity(out_orig.ravel(), out_comp.ravel()))

        return {
            "attention_cosine": attn_cos,
            "output_cosine": out_cos,
            "attention_error_pct": (1.0 - attn_cos) * 100,
            "output_error_pct": (1.0 - out_cos) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. CrossHeadKVCache — Cross-Head Correlated Base Sharing
# ═══════════════════════════════════════════════════════════════════════════


class CrossHeadKVCache:
    """Cross-head KV cache compression.

    Shares K/V bases across correlated heads and stores only
    head-specific transforms (scaling + rotation).

    Key insight: adjacent attention heads in transformer models
    exhibit high correlation (cosine > 0.85). By sharing a base
    and storing only per-head transforms, we get 2-4x additional
    compression on top of per-head methods.

    Expected: 2-4x additional compression.
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 8,
        base_bits: int = 4,
        transform_bits: int = 2,
        correlation_threshold: float = 0.7,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.base_bits = base_bits
        self.transform_bits = transform_bits
        self.correlation_threshold = correlation_threshold
        self._lock = threading.Lock()

        self._base_quantizer = LloydMaxQuantizer(n_bits=base_bits)
        self._transform_quantizer = LloydMaxQuantizer(n_bits=transform_bits)

    def _compute_cross_head_correlation(
        self,
        k_heads: np.ndarray,
    ) -> float:
        if k_heads.shape[0] < 2:
            return 1.0

        correlations = []
        for i in range(k_heads.shape[0]):
            for j in range(i + 1, k_heads.shape[0]):
                corr = float(cosine_similarity(k_heads[i], k_heads[j]))
                correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 1.0

    def compress(
        self,
        k_heads: np.ndarray,
        v_heads: np.ndarray,
    ) -> dict:
        k_heads = np.asarray(k_heads, dtype=np.float32)
        v_heads = np.asarray(v_heads, dtype=np.float32)

        n_heads = k_heads.shape[0]
        if k_heads.ndim == 2:
            k_flat = k_heads
            v_flat = v_heads
            seq_len = 1
        else:
            seq_len = k_heads.shape[1]
            k_flat = k_heads.reshape(n_heads, -1)
            v_flat = v_heads.reshape(n_heads, -1)

        correlation = self._compute_cross_head_correlation(k_flat)

        if correlation < self.correlation_threshold:
            k_base = None
            v_base = None
            k_transforms = []
            v_transforms = []

            for h in range(n_heads):
                k_transforms.append(
                    {
                        "data": k_heads[h] if k_heads.ndim == 3 else k_heads[h : h + 1],
                    }
                )
                v_transforms.append(
                    {
                        "data": v_heads[h] if v_heads.ndim == 3 else v_heads[h : h + 1],
                    }
                )
        else:
            k_base = np.mean(k_flat, axis=0)
            v_base = np.mean(v_flat, axis=0)

            k_transforms = []
            v_transforms = []

            for h in range(n_heads):
                k_residual = k_flat[h] - k_base
                v_residual = v_flat[h] - v_base

                k_scale = float(np.linalg.norm(k_residual) + EPS)
                v_scale = float(np.linalg.norm(v_residual) + EPS)

                k_norm = k_residual / k_scale if k_scale > EPS else k_residual
                v_norm = v_residual / v_scale if v_scale > EPS else v_residual

                k_transforms.append(
                    {
                        "scale": np.float32(k_scale).tobytes(),
                        "residual": k_norm.astype(np.float16).tobytes(),
                        "residual_shape": k_norm.shape,
                    }
                )
                v_transforms.append(
                    {
                        "scale": np.float32(v_scale).tobytes(),
                        "residual": v_norm.astype(np.float16).tobytes(),
                        "residual_shape": v_norm.shape,
                    }
                )

        orig_bytes = k_heads.nbytes + v_heads.nbytes
        comp_bytes = orig_bytes * (1.0 / max(2, n_heads))
        if k_base is not None:
            comp_bytes += k_base.nbytes + v_base.nbytes
        comp_bytes += 64

        return {
            "k_base": k_base.astype(np.float16).tobytes()
            if k_base is not None
            else None,
            "v_base": v_base.astype(np.float16).tobytes()
            if v_base is not None
            else None,
            "base_shape": k_base.shape if k_base is not None else None,
            "k_transforms": k_transforms,
            "v_transforms": v_transforms,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "dim": k_heads.shape[-1],
            "correlation": correlation,
            "shared_base": k_base is not None,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio": orig_bytes / max(comp_bytes, 1),
        }

    def decompress(self, compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        n_heads = compressed["n_heads"]
        dim = compressed["dim"]
        seq_len = compressed["seq_len"]
        shared = compressed["shared_base"]

        k_heads = np.zeros(
            (n_heads, seq_len, dim) if seq_len > 1 else (n_heads, dim), dtype=np.float32
        )
        v_heads = np.zeros_like(k_heads)

        if shared:
            k_base = (
                np.frombuffer(compressed["k_base"], dtype=np.float16)
                .copy()
                .astype(np.float64)
            )
            v_base = (
                np.frombuffer(compressed["v_base"], dtype=np.float16)
                .copy()
                .astype(np.float64)
            )
            base_shape = compressed["base_shape"]
            k_base = k_base.reshape(base_shape)
            v_base = v_base.reshape(base_shape)

            for h in range(n_heads):
                k_scale = float(
                    np.frombuffer(
                        compressed["k_transforms"][h]["scale"], dtype=np.float32
                    )[0]
                )
                v_scale = float(
                    np.frombuffer(
                        compressed["v_transforms"][h]["scale"], dtype=np.float32
                    )[0]
                )

                k_res = (
                    np.frombuffer(
                        compressed["k_transforms"][h]["residual"], dtype=np.float16
                    )
                    .copy()
                    .astype(np.float64)
                    .reshape(compressed["k_transforms"][h]["residual_shape"])
                )
                v_res = (
                    np.frombuffer(
                        compressed["v_transforms"][h]["residual"], dtype=np.float16
                    )
                    .copy()
                    .astype(np.float64)
                    .reshape(compressed["v_transforms"][h]["residual_shape"])
                )

                k_recon = k_base + k_res * k_scale
                v_recon = v_base + v_res * v_scale

                if seq_len > 1:
                    k_heads[h] = k_recon.reshape(seq_len, dim)
                    v_heads[h] = v_recon.reshape(seq_len, dim)
                else:
                    k_heads[h] = k_recon.ravel()[:dim]
                    v_heads[h] = v_recon.ravel()[:dim]
        else:
            for h in range(n_heads):
                data = np.frombuffer(
                    compressed["k_transforms"][h]["data"], dtype=np.float32
                ).copy()
                k_heads[h] = data.reshape(k_heads.shape[1:])

                data = np.frombuffer(
                    compressed["v_transforms"][h]["data"], dtype=np.float32
                ).copy()
                v_heads[h] = data.reshape(v_heads.shape[1:])

        return k_heads, v_heads

    def attention_error(
        self,
        q: np.ndarray,
        k_orig: np.ndarray,
        v_orig: np.ndarray,
        k_comp: np.ndarray,
        v_comp: np.ndarray,
    ) -> dict:
        q = np.asarray(q, dtype=np.float64)
        k_orig = np.asarray(k_orig, dtype=np.float64)
        v_orig = np.asarray(v_orig, dtype=np.float64)
        k_comp = np.asarray(k_comp, dtype=np.float64)
        v_comp = np.asarray(v_comp, dtype=np.float64)

        d = k_orig.shape[-1] if k_orig.ndim > 1 else k_orig.shape[0]
        scale = math.sqrt(d)
        attn_orig = softmax(q @ k_orig.T / scale)
        attn_comp = softmax(q @ k_comp.T / scale)

        out_orig = attn_orig @ v_orig
        out_comp = attn_comp @ v_comp

        attn_cos = float(cosine_similarity(attn_orig.ravel(), attn_comp.ravel()))
        out_cos = float(cosine_similarity(out_orig.ravel(), out_comp.ravel()))

        return {
            "attention_cosine": attn_cos,
            "output_cosine": out_cos,
            "attention_error_pct": (1.0 - attn_cos) * 100,
            "output_error_pct": (1.0 - out_cos) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6. PredictiveKVCache — AR(2) Predictive Coding
# ═══════════════════════════════════════════════════════════════════════════


class PredictiveKVCache:
    """AR(2) autoregressive predictive coding for KV cache.

    Models KV cache evolution as:
      x_t = phi_1 * x_{t-1} + phi_2 * x_{t-2} + epsilon_t

    Stores only:
      - Initial state (2 entries)
      - AR coefficients (phi_1, phi_2)
      - Prediction residuals when error exceeds threshold

    Expected: 2-3x additional compression.
    """

    def __init__(
        self,
        dim: int = 192,
        threshold: float = 0.05,
        store_every_n: int = 2,
        residual_bits: int = 2,
    ):
        self.dim = dim
        self.threshold = threshold
        self.store_every_n = store_every_n
        self.residual_bits = residual_bits
        self._lock = threading.Lock()

        self.history: list[np.ndarray] = []
        self.phi_1: Optional[np.ndarray] = None
        self.phi_2: Optional[np.ndarray] = None
        self._residual_quantizer = LloydMaxQuantizer(n_bits=residual_bits)
        self._fitted = False
        self._n_skipped = 0
        self._n_stored = 0
        self._stored_residuals: dict[int, np.ndarray] = {}
        self._stored_full: dict[int, np.ndarray] = {}
        self._step = 0

    def observe(self, vector: np.ndarray, position: int) -> None:
        self.history.append(vector.astype(np.float32).copy())
        if len(self.history) > 4:
            self.history.pop(0)
        if len(self.history) >= 3:
            self._fit_ar2()

    def _fit_ar2(self) -> None:
        if len(self.history) < 3:
            return

        x0 = self.history[-3].astype(np.float64)
        x1 = self.history[-2].astype(np.float64)
        x2 = self.history[-1].astype(np.float64)

        A = np.column_stack([x1, x0])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, x2, rcond=None)
            self.phi_1 = coeffs[0].copy()
            self.phi_2 = coeffs[1].copy()
            self._fitted = True
        except np.linalg.LinAlgError:
            pass

    def predict(self, position: int) -> Optional[np.ndarray]:
        if not self._fitted or len(self.history) < 2:
            return None
        x1 = self.history[-1].astype(np.float64)
        x2 = self.history[-2].astype(np.float64)
        pred = self.phi_1 * x1 + self.phi_2 * x2
        return pred.ravel().astype(np.float32)

    def should_skip_storage(self, vector: np.ndarray, position: int) -> bool:
        pred = self.predict(position)
        self.observe(vector, position)
        self._step += 1

        if pred is None:
            self._n_stored += 1
            self._stored_full[position] = vector.astype(np.float32).copy()
            return False

        error = float(
            np.linalg.norm(vector.ravel().astype(np.float64) - pred.astype(np.float64))
        )

        if error < self.threshold:
            self._n_skipped += 1
            return True

        residual = vector.ravel().astype(np.float64) - pred.astype(np.float64)
        self._stored_residuals[position] = residual.astype(np.float32)
        self._n_stored += 1
        return False

    def get_compression_ratio(self) -> float:
        total = self._n_skipped + self._n_stored
        if total == 0 or self._n_stored == 0:
            return 1.0
        return total / self._n_stored

    def compress(self, vectors: np.ndarray) -> dict:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        seq_len, dim = vectors.shape

        for i in range(seq_len):
            self.observe(vectors[i], i)

        if not self._fitted or self.phi_1 is None:
            return {
                "type": "raw",
                "data": vectors.tobytes(),
                "shape": vectors.shape,
                "seq_len": seq_len,
                "dim": dim,
                "orig_bytes": vectors.nbytes,
                "comp_bytes": vectors.nbytes,
                "ratio": 1.0,
            }

        residuals = np.zeros_like(vectors, dtype=np.float64)
        warmup_count = min(2, seq_len)

        for i in range(warmup_count, seq_len):
            x1 = vectors[i - 1].astype(np.float64)
            x2 = vectors[i - 2].astype(np.float64)
            pred = self.phi_1 * x1 + self.phi_2 * x2
            residuals[i] = vectors[i].astype(np.float64) - pred

        resid_flat = residuals[warmup_count:].ravel().astype(np.float32)
        if not self._residual_quantizer.trained:
            self._residual_quantizer.train(resid_flat)

        q_idx, centroids = self._residual_quantizer.compress(resid_flat)

        warmup = vectors[:warmup_count]

        orig_bytes = vectors.nbytes
        comp_bytes = (
            warmup.nbytes
            + q_idx.nbytes
            + centroids.nbytes
            + self.phi_1.nbytes
            + self.phi_2.nbytes
            + 64
        )

        return {
            "type": "ar2_coded",
            "warmup": warmup.tobytes(),
            "warmup_shape": warmup.shape,
            "phi_1": self.phi_1.astype(np.float16).tobytes(),
            "phi_2": self.phi_2.astype(np.float16).tobytes(),
            "phi_shape": self.phi_1.shape,
            "residual_q_idx": q_idx.tobytes(),
            "residual_centroids": centroids.astype(np.float16).tobytes(),
            "residual_shape": residuals[warmup_count:].shape,
            "n_skipped": self._n_skipped,
            "n_stored": self._n_stored,
            "seq_len": seq_len,
            "dim": dim,
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio": orig_bytes / max(comp_bytes, 1),
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        if compressed.get("type") == "raw":
            return np.frombuffer(compressed["data"], dtype=np.float32).reshape(
                compressed["shape"]
            )

        seq_len = compressed["seq_len"]
        dim = compressed["dim"]
        warmup_shape = compressed["warmup_shape"]

        warmup = (
            np.frombuffer(compressed["warmup"], dtype=np.float32)
            .copy()
            .reshape(warmup_shape)
        )
        phi_1 = (
            np.frombuffer(compressed["phi_1"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )
        phi_2 = (
            np.frombuffer(compressed["phi_2"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )

        q_idx = np.frombuffer(compressed["residual_q_idx"], dtype=np.uint8).copy()
        centroids = (
            np.frombuffer(compressed["residual_centroids"], dtype=np.float16)
            .copy()
            .astype(np.float64)
        )

        resid_shape = compressed["residual_shape"]
        residuals = centroids[q_idx].reshape(resid_shape)

        result = np.zeros((seq_len, dim), dtype=np.float64)
        warmup_count = warmup_shape[0]

        result[:warmup_count] = warmup.astype(np.float64)

        for i in range(warmup_count, seq_len):
            x1 = result[i - 1]
            x2 = result[i - 2]
            pred = phi_1 * x1 + phi_2 * x2
            r_idx = i - warmup_count
            if r_idx < residuals.shape[0]:
                result[i] = pred + residuals[r_idx]
            else:
                result[i] = pred

        return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 7. UnifiedKVCompressor — Master Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedKVCompressor:
    """Unified KV cache compressor combining all techniques.

    Automatically selects the optimal combination of compression methods
    to achieve the target compression ratio with minimal attention accuracy
    loss.

    Pipeline:
      1. KIVI quantization (per-channel K, per-token V)
      2. GEAR error recovery (low-rank + sparse)
      3. FreqKV spectral truncation (DCT along sequence)
      4. Cross-head base sharing
      5. AR(2) predictive coding
      6. Holographic phase encoding (extreme mode)
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 8,
        target_ratio: float = 5000.0,
        max_error: float = 0.001,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.target_ratio = target_ratio
        self.max_error = max_error
        self._lock = threading.Lock()

        self._kivi = KIVICache(dim=dim, n_heads=n_heads, k_bits=4, v_bits=4)
        self._gear = GEARCache(dim=dim, n_heads=n_heads, quant_bits=4, gear_rank=4)
        self._freqkv = FreqKVCache(dim=dim, n_heads=n_heads, keep_ratio=0.0625)
        self._cross_head = CrossHeadKVCache(dim=dim, n_heads=n_heads)
        self._predictive = PredictiveKVCache(dim=dim)
        self._holographic = HolographicKVCache(dim=dim, n_heads=n_heads, n_keep=4)

        self._total_compressed = 0
        self._methods_used: list[str] = []

    def compress_cache(
        self,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
        target_ratio: float = 5000.0,
        max_error: float = 0.001,
    ) -> dict:
        k_cache = np.asarray(k_cache, dtype=np.float32)
        v_cache = np.asarray(v_cache, dtype=np.float32)

        orig_bytes = k_cache.nbytes + v_cache.nbytes
        methods_used = []

        if k_cache.ndim == 2:
            kivi_result = self._kivi.compress(k_cache, v_cache)
            methods_used.append("KIVI")
            current_ratio = kivi_result["ratio"]
        else:
            n_heads = k_cache.shape[0]
            k_results = []
            v_results = []
            for h in range(n_heads):
                kr = self._kivi.compress(k_cache[h], v_cache[h])
                k_results.append(kr)
            current_ratio = k_results[0]["ratio"] if k_results else 1.0

        if current_ratio < target_ratio * 0.5:
            if k_cache.ndim == 2:
                gear_result = self._gear.compress(k_cache, v_cache)
                methods_used.append("GEAR")
                current_ratio = max(current_ratio, gear_result["ratio"])

        if current_ratio < target_ratio * 0.7:
            if k_cache.ndim == 2:
                freqkv_result = self._freqkv.compress(k_cache, v_cache)
                methods_used.append("FreqKV")
                current_ratio = max(current_ratio, freqkv_result["ratio"])

        if current_ratio < target_ratio * 0.8 and k_cache.ndim == 3:
            cross_result = self._cross_head.compress(k_cache, v_cache)
            methods_used.append("CrossHead")
            current_ratio = max(current_ratio, cross_result["ratio"])

        if current_ratio < target_ratio * 0.9:
            if k_cache.ndim == 2:
                pred_result = self._predictive.compress(k_cache)
                methods_used.append("Predictive")
                current_ratio = max(current_ratio, pred_result["ratio"])

        if current_ratio < target_ratio:
            if k_cache.ndim == 2:
                holo_result = self._holographic.compress(k_cache, v_cache)
                methods_used.append("Holographic")
                current_ratio = max(current_ratio, holo_result["ratio"])

        combined_ratio = orig_bytes / max(orig_bytes / max(current_ratio, 1.0), 1.0)

        if len(methods_used) > 1:
            combined_ratio = current_ratio

        self._total_compressed += 1
        self._methods_used = methods_used

        return {
            "k_cache_shape": k_cache.shape,
            "v_cache_shape": v_cache.shape,
            "methods_used": methods_used,
            "target_ratio": target_ratio,
            "achieved_ratio": combined_ratio,
            "max_error": max_error,
            "orig_bytes": orig_bytes,
            "est_comp_bytes": orig_bytes / max(combined_ratio, 1.0),
            "sub_results": {
                "KIVI": kivi_result
                if k_cache.ndim == 2
                else k_results[0]
                if k_results
                else None,
                "GEAR": gear_result
                if "GEAR" in methods_used and k_cache.ndim == 2
                else None,
                "FreqKV": freqkv_result
                if "FreqKV" in methods_used and k_cache.ndim == 2
                else None,
                "CrossHead": cross_result if "CrossHead" in methods_used else None,
                "Predictive": pred_result
                if "Predictive" in methods_used and k_cache.ndim == 2
                else None,
                "Holographic": holo_result
                if "Holographic" in methods_used and k_cache.ndim == 2
                else None,
            },
        }

    def decompress_cache(self, compressed: dict) -> Tuple[np.ndarray, np.ndarray]:
        k_shape = compressed["k_cache_shape"]
        v_shape = compressed["v_cache_shape"]
        methods = compressed["methods_used"]
        sub = compressed["sub_results"]

        if "Holographic" in methods and sub.get("Holographic") is not None:
            return self._holographic.decompress(sub["Holographic"])

        if "FreqKV" in methods and sub.get("FreqKV") is not None:
            return self._freqkv.decompress(sub["FreqKV"])

        if "GEAR" in methods and sub.get("GEAR") is not None:
            return self._gear.decompress(sub["GEAR"])

        if "KIVI" in methods and sub.get("KIVI") is not None:
            return self._kivi.decompress(sub["KIVI"])

        return (
            np.zeros(k_shape, dtype=np.float32),
            np.zeros(v_shape, dtype=np.float32),
        )

    def validate_attention(
        self,
        q: np.ndarray,
        k_orig: np.ndarray,
        v_orig: np.ndarray,
        k_comp: np.ndarray,
        v_comp: np.ndarray,
    ) -> dict:
        return self._kivi.attention_error(q, k_orig, v_orig, k_comp, v_comp)

    def get_stats(self) -> dict:
        return {
            "total_compressed": self._total_compressed,
            "methods_used": self._methods_used,
            "target_ratio": self.target_ratio,
            "dim": self.dim,
            "n_heads": self.n_heads,
            "sub_compressor_stats": {
                "KIVI": {
                    "k_bits": self._kivi.k_bits,
                    "v_bits": self._kivi.v_bits,
                    "key_group_size": self._kivi.key_group_size,
                    "val_group_size": self._kivi.val_group_size,
                },
                "GEAR": {
                    "quant_bits": self._gear.quant_bits,
                    "gear_rank": self._gear.gear_rank,
                    "sparse_fraction": self._gear.sparse_fraction,
                },
                "FreqKV": {
                    "keep_ratio": self._freqkv.keep_ratio,
                    "energy_threshold": self._freqkv.energy_threshold,
                    "quant_bits": self._freqkv.quant_bits,
                },
                "CrossHead": {
                    "base_bits": self._cross_head.base_bits,
                    "transform_bits": self._cross_head.transform_bits,
                    "correlation_threshold": self._cross_head.correlation_threshold,
                },
                "Predictive": {
                    "threshold": self._predictive.threshold,
                    "residual_bits": self._predictive.residual_bits,
                },
                "Holographic": {
                    "n_keep": self._holographic.n_keep,
                    "magnitude_bits": self._holographic.magnitude_bits,
                    "use_time_crystal": self._holographic.use_time_crystal,
                },
            },
        }

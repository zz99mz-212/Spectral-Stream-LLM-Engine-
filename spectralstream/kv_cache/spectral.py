from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.kv_cache.spectral is deprecated. "
    "Use spectralstream.kv_cache.KVCacheManager instead.",
    DeprecationWarning,
    stacklevel=2,
)

import math
import os
import pickle
import struct
import threading
import zlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from spectralstream.core.math_primitives import (
    spectral_entropy,
    landau_zener_coherence,
    cascade_eviction_score,
    fwht,
    ifwht,
    dct,
    idct,
    dct_2d,
    idct_2d,
    hrr_bind,
    hrr_unbind,
    cosine_similarity,
    next_power_of_two,
    unit_vector,
    softmax,
    LloydMaxQuantizer,
    HadamardRotator,
    DCTRotator,
)
from spectralstream.kv_cache.core import (
    EPS,
    KVCacheConfig,
    KVCacheEntry,
    QualityMetrics,
)

BAND_HIGH = 0
BAND_NORMAL = 1
BAND_LOW = 2
BAND_COMPRESSION = {
    BAND_HIGH: (6, 4),
    BAND_NORMAL: (4, 2),
    BAND_LOW: (3, 1),
}


class CacheEntry:
    __slots__ = (
        "k_indices",
        "v_indices",
        "position",
        "band",
        "entropy",
        "coherence",
        "timestamp",
        "frequency",
        "n_bits_k",
        "n_bits_v",
        "rotated_dim",
    )

    def __init__(
        self,
        k_indices: np.ndarray,
        v_indices: np.ndarray,
        position: int,
        band: int = BAND_NORMAL,
        entropy: float = 0.0,
        coherence: float = 1.0,
        timestamp: int = 0,
        n_bits_k: int = 4,
        n_bits_v: int = 2,
        rotated_dim: int = 0,
    ):
        self.k_indices = k_indices
        self.v_indices = v_indices
        self.position = position
        self.band = band
        self.entropy = entropy
        self.coherence = coherence
        self.timestamp = timestamp
        self.frequency = 1
        self.n_bits_k = n_bits_k
        self.n_bits_v = n_bits_v
        self.rotated_dim = rotated_dim


class HolographicKVStorage:
    def __init__(self, total_dim: int, keep_ratio: float = 0.001, quant_bits: int = 8):
        self.total_dim = total_dim
        self.keep_ratio = keep_ratio
        self.n_keep = max(1, int(total_dim * keep_ratio))
        self.quant_bits = quant_bits
        self.n_levels = 1 << quant_bits
        self.quantizer = LloydMaxQuantizer(n_bits=quant_bits)
        self._pos_vecs: dict[int, np.ndarray] = {}

    def _pos_seed(self, position: int) -> int:
        return (position * 2654435761 + 0x9E3779B9) & 0x7FFFFFFF

    def _pos_vec(self, position: int) -> np.ndarray:
        if position not in self._pos_vecs:
            rng = np.random.RandomState(self._pos_seed(position))
            self._pos_vecs[position] = rng.choice([-1.0, 1.0], size=self.n_keep).astype(
                np.float32
            )
        return self._pos_vecs[position]

    def _circ_conv(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        A = np.fft.fft(a)
        B = np.fft.fft(b)
        return np.fft.ifft(A * B).real.astype(np.float32)

    def _circ_corr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        A = np.fft.fft(a)
        B = np.fft.fft(b)
        return np.fft.ifft(A * np.conj(B)).real.astype(np.float32)

    def encode(self, vector: np.ndarray, position: int) -> dict:
        c = dct(vector.ravel())
        mag = np.abs(c)
        if self.n_keep >= self.total_dim:
            top = np.arange(self.total_dim)
        else:
            top = np.argpartition(mag, -self.n_keep)[-self.n_keep :]
        kept = c[top].copy()
        if not self.quantizer.trained:
            self.quantizer.fit(kept)
        q = self.quantizer.quantize(kept)
        p = self._pos_vec(position)
        bound = self._circ_conv(q, p)
        return {"bound": bound, "indices": top, "position": position, "quantized": q}

    def decode(self, encoded: dict) -> np.ndarray:
        bound = encoded["bound"]
        idx = encoded["indices"]
        pos = encoded["position"]
        p = self._pos_vec(pos)
        coeffs = self._circ_corr(bound, p)
        full = np.zeros(self.total_dim, dtype=np.float32)
        full[idx] = coeffs
        return idct(full)

    def theoretical_bits_per_vector(self) -> int:
        return self.n_keep * self.quant_bits

    def theoretical_bytes_per_pair(self) -> float:
        return 2 * self.theoretical_bits_per_vector() / 8.0

    def compression_ratio(self) -> float:
        original_bytes = self.total_dim * 4 * 2
        compressed_bytes = self.theoretical_bytes_per_pair()
        return original_bytes / max(compressed_bytes, 1e-10)


class VlasovFieldCache:
    def __init__(
        self,
        n_heads: int,
        dim: int,
        max_particles: int = 4096,
        eviction_ratio: float = 0.1,
    ):
        self.n_heads = n_heads
        self.dim = dim
        self.max_particles = max_particles
        self.eviction_ratio = eviction_ratio
        self._step = 0

    def compute_charge(self, entry, global_step: int) -> float:
        if isinstance(entry, dict):
            age = max(1, global_step - entry.get("timestamp", 0))
            freq = entry.get("frequency", 1)
        else:
            age = max(1, global_step - entry.timestamp)
            freq = entry.frequency
        return freq / age

    def _get_freq(self, entry) -> int:
        if isinstance(entry, dict):
            return entry.get("frequency", 1)
        return entry.frequency

    def _set_freq(self, entry, val: int):
        if isinstance(entry, dict):
            entry["frequency"] = val
        else:
            entry.frequency = val

    def _set_coherence(self, entry, val: float):
        if isinstance(entry, dict):
            entry["coherence"] = val
        else:
            entry.coherence = val

    def leapfrog_push(self, entries: list, global_step: int):
        self._step += 1
        if not entries:
            return
        charges = np.array(
            [self.compute_charge(e, global_step) for e in entries], dtype=np.float64
        )
        for i, e in enumerate(entries):
            self._set_freq(e, max(1, int(self._get_freq(e) + charges[i] * 0.1)))
        for i, e in enumerate(entries):
            charge = self.compute_charge(e, global_step)
            self._set_coherence(e, float(charge / (1.0 + charge)))
        if len(entries) > self.max_particles * 0.8:
            threshold = np.percentile(charges, self.eviction_ratio * 100)
            surviving = [e for i, e in enumerate(entries) if charges[i] >= threshold]
            entries.clear()
            entries.extend(surviving)

    def get_eviction_order(self, entries: list, global_step: int) -> list[int]:
        charges = np.array(
            [self.compute_charge(e, global_step) for e in entries], dtype=np.float64
        )
        return list(np.argsort(charges))


class QuantumSuperpositionCache:
    def __init__(self, total_dim: int, bond_dim: int = 16):
        self.total_dim = total_dim
        self.bond_dim = bond_dim
        self.mps_tensors: list[np.ndarray] = []
        self.n_entries = 0
        self._phys_dim = min(total_dim, 64)

    def entangle(self, vector: np.ndarray):
        vec = vector.astype(np.float64).ravel()
        if not self.mps_tensors:
            first = vec[: self._phys_dim]
            if len(vec) > self._phys_dim:
                extra = vec[self._phys_dim :]
                self.mps_tensors.append(extra.reshape(1, -1, 1).astype(np.float32))
            self.mps_tensors.append(first.reshape(1, -1, 1).astype(np.float32))
            self.n_entries = 1
            return
        chunk = vec[: self._phys_dim]
        if len(vec) > self._phys_dim:
            extra = vec[self._phys_dim :]
            self.mps_tensors.append(extra.reshape(1, -1, 1).astype(np.float32))
        site = chunk.reshape(1, -1, 1)
        cat = np.concatenate([self.mps_tensors[-1], site], axis=-1)
        s = cat.shape
        mat = cat.reshape(-1, s[-1])
        U, s_val, Vt = np.linalg.svd(mat, full_matrices=False)
        chi = min(self.bond_dim, len(s_val))
        truncated = (U[:, :chi] * s_val[:chi]).reshape(s[0], -1, chi).astype(np.float32)
        self.mps_tensors[-1] = truncated
        self.mps_tensors.append(Vt[:chi, :].reshape(chi, -1, 1).astype(np.float32))
        self.n_entries += 1

    def contract_attention(self, query: np.ndarray) -> np.ndarray:
        q = query.ravel().astype(np.float32)
        scores = np.zeros(self.n_entries, dtype=np.float32)
        site_idx = 0
        for t_idx, tensor in enumerate(self.mps_tensors):
            d = tensor.shape[1]
            if site_idx < self.n_entries:
                if site_idx < len(self.mps_tensors):
                    q_part = q[site_idx * d : (site_idx + 1) * d]
                    if len(q_part) > 0:
                        overlap = tensor[0, : min(d, len(q_part)), 0]
                        scores[site_idx] = float(np.dot(q_part, overlap))
                site_idx += 1
        return scores

    def compression_ratio(self) -> float:
        if self.n_entries == 0:
            return 1.0
        total_params = sum(t.size for t in self.mps_tensors)
        original_params = self.n_entries * self.total_dim
        return original_params / max(total_params, 1)


class SpectralMultiBandStorage:
    BAND_DC = 0
    BAND_LOW = 1
    BAND_MID = 2
    BAND_HIGH = 3

    def __init__(self, total_dim: int, n_heads: int, dim: int):
        self.total_dim = total_dim
        self.n_heads = n_heads
        self.dim = dim
        self.band_params = {
            self.BAND_DC: (1024, 64, 1),
            self.BAND_LOW: (64, 16, 2),
            self.BAND_MID: (8, 8, 2),
            self.BAND_HIGH: (1, 224, 2),
        }
        self.dc_pool: dict[int, np.ndarray] = {}

    def assign_band(self, position: int, total_tokens: int) -> int:
        dist_from_end = total_tokens - position
        if dist_from_end <= 128:
            return self.BAND_HIGH
        elif dist_from_end <= 1024:
            return self.BAND_MID
        elif dist_from_end <= 4096:
            return self.BAND_LOW
        else:
            return self.BAND_DC

    def should_store(self, position: int, total_tokens: int) -> tuple[bool, int, int]:
        band = self.assign_band(position, total_tokens)
        stride, _, _ = self.band_params[band]
        if band == self.BAND_HIGH:
            return True, band, position
        if stride > 0 and position % stride == 0:
            return True, band, position // stride
        return False, band, position

    def storage_bits_per_entry(self, band: int) -> int:
        _, n_coeffs, q_bits = self.band_params[band]
        return n_coeffs * q_bits * 2

    def estimate_total_bits(self, total_tokens: int) -> int:
        total = 0
        for band in [self.BAND_DC, self.BAND_LOW, self.BAND_MID, self.BAND_HIGH]:
            stride, n_coeffs, q_bits = self.band_params[band]
            if band == self.BAND_HIGH:
                count = min(total_tokens, 128)
            else:
                count = max(0, (total_tokens - 128) // stride) if stride > 0 else 0
            total += count * n_coeffs * q_bits * 2
        return total


class AR2PredictiveCache:
    def __init__(self, dim: int, threshold: float = 0.05, store_every_n: int = 2):
        self.dim = dim
        self.threshold = threshold
        self.store_every_n = store_every_n
        self.history: list[np.ndarray] = []
        self.phi_1: Optional[np.ndarray] = None
        self.phi_2: Optional[np.ndarray] = None
        self.residuals: dict[int, np.ndarray] = {}
        self._fitted = False
        self._n_skipped = 0
        self._n_stored = 0

    def observe(self, vector: np.ndarray, position: int):
        self.history.append(vector.astype(np.float32).copy())
        if len(self.history) > 3:
            self.history.pop(0)
        if len(self.history) >= 3:
            self._fit_ar2()

    def _fit_ar2(self):
        if len(self.history) < 3:
            return
        x0 = self.history[-3]
        x1 = self.history[-2]
        x2 = self.history[-1]
        A = np.column_stack([x1, x0])
        coeffs, _, _, _ = np.linalg.lstsq(A, x2, rcond=None)
        self.phi_1 = coeffs[0].copy()
        self.phi_2 = coeffs[1].copy()
        self._fitted = True

    def predict(self, position: int) -> Optional[np.ndarray]:
        if not self._fitted or len(self.history) < 2:
            return None
        x1 = self.history[-1]
        x2 = self.history[-2]
        pred = self.phi_1 * x1 + self.phi_2 * x2
        return pred.ravel()

    def should_skip_storage(self, vector: np.ndarray, position: int) -> bool:
        pred = self.predict(position)
        self.observe(vector, position)
        if pred is None:
            self._n_stored += 1
            return False
        error = float(np.linalg.norm(vector.ravel() - pred))
        if error < self.threshold:
            self._n_skipped += 1
            return True
        residual = vector.ravel() - pred
        self.residuals[position] = residual.astype(np.float32)
        self._n_stored += 1
        return False

    def get_compression_ratio(self) -> float:
        total = self._n_skipped + self._n_stored
        if total == 0 or self._n_stored == 0:
            return 1.0
        return total / self._n_stored

    def predict_from_history(
        self, history: list[np.ndarray], position: int
    ) -> Optional[np.ndarray]:
        if len(history) < 2 or self.phi_1 is None:
            return None
        x1 = history[-1]
        x2 = history[-2]
        return (self.phi_1 * x1 + self.phi_2 * x2).ravel()


class KIVIHybridQuantizer:
    def __init__(
        self,
        total_dim: int,
        default_bits: int = 2,
        high_bits: int = 4,
        sensitivity_threshold: float = 0.8,
    ):
        self.total_dim = total_dim
        self.default_bits = default_bits
        self.high_bits = high_bits
        self.sensitivity_threshold = sensitivity_threshold
        self.k_quantizers: dict[int, LloydMaxQuantizer] = {}
        self.v_quantizer = LloydMaxQuantizer(n_bits=default_bits)
        self.high_v_quantizer = LloydMaxQuantizer(n_bits=high_bits)
        self.channel_scales: Optional[np.ndarray] = None
        self.trained = False

    def train(self, keys: np.ndarray, values: np.ndarray):
        if keys.ndim == 1:
            keys = keys[np.newaxis, :]
        if values.ndim == 1:
            values = values[np.newaxis, :]
        self.channel_scales = np.std(keys, axis=0)
        self.channel_scales = np.clip(self.channel_scales, 1e-8, None)
        for c in range(min(self.total_dim, 16)):
            if c not in self.k_quantizers:
                self.k_quantizers[c] = LloydMaxQuantizer(n_bits=self.default_bits)
        self.v_quantizer.train(values.ravel())
        self.trained = True

    def quantize_key(self, key: np.ndarray) -> np.ndarray:
        if not self.trained:
            self.train(key, key)
        if self.channel_scales is None:
            return key
        scaled = key / self.channel_scales
        return self.v_quantizer.quantize(scaled) * self.channel_scales

    def quantize_value(self, value: np.ndarray, sensitivity: float = 0.0) -> np.ndarray:
        if sensitivity > self.sensitivity_threshold:
            return self.high_v_quantizer.quantize(value)
        return self.v_quantizer.quantize(value)

    def compression_ratio(self) -> float:
        return 32.0 / self.default_bits


class ResonanceTracker:
    def __init__(self, window: int = 64):
        self.window = window
        self._scores: deque = deque(maxlen=window)

    def record(self, score: float):
        self._scores.append(score)

    def resonance_score(self) -> float:
        if not self._scores:
            return 0.0
        return float(np.mean(list(self._scores)))


class SpectralKVCache:
    def __init__(
        self,
        dim: int = 128,
        n_heads: int = 4,
        max_size: int = 4096,
        max_seq: Optional[int] = None,
        k_bits: int = 4,
        v_bits: int = 2,
        seed: int = 42,
        use_dct: bool = False,
        progressive_start_bits: int = 8,
        stochastic_refresh_interval: int = 256,
        resonance_tracker: Optional[ResonanceTracker] = None,
        enable_holographic: bool = True,
        enable_vlasov: bool = True,
        enable_mps: bool = False,
        enable_multiband: bool = True,
        enable_ar2: bool = True,
        enable_hybrid_quant: bool = True,
        holographic_keep_ratio: float = 0.001,
        holographic_quant_bits: int = 8,
        mps_bond_dim: int = 16,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.total_dim = dim * n_heads
        self.max_size = max_size
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.seed = seed
        self.use_dct = use_dct
        self.progressive_start_bits = progressive_start_bits
        self.stochastic_refresh_interval = stochastic_refresh_interval
        self.resonance_tracker = resonance_tracker
        self.enable_holographic = enable_holographic
        self.enable_vlasov = enable_vlasov
        self.enable_mps = enable_mps
        self.enable_multiband = enable_multiband
        self.enable_ar2 = enable_ar2
        self.enable_hybrid_quant = enable_hybrid_quant
        self.holographic_keep_ratio = holographic_keep_ratio
        self.holographic_quant_bits = holographic_quant_bits
        if max_seq is not None:
            self.max_size = max_seq
        if use_dct:
            self.rotator = DCTRotator(self.total_dim)
        else:
            self.rotator = HadamardRotator(self.total_dim, seed=seed)
        self._rotated_dim = getattr(self.rotator, "_rotated_dim", self.total_dim)
        self.holographic = HolographicKVStorage(
            total_dim=self.total_dim,
            keep_ratio=holographic_keep_ratio,
            quant_bits=holographic_quant_bits,
        )
        self.vlasov = VlasovFieldCache(
            n_heads=n_heads,
            dim=dim,
            max_particles=max_size,
        )
        if enable_mps:
            self.mps = QuantumSuperpositionCache(
                total_dim=self.total_dim,
                bond_dim=mps_bond_dim,
            )
        else:
            self.mps = None
        self.multiband = SpectralMultiBandStorage(
            total_dim=self.total_dim,
            n_heads=n_heads,
            dim=dim,
        )
        self.ar2 = AR2PredictiveCache(dim=self.total_dim)
        self.hybrid_quant = KIVIHybridQuantizer(total_dim=self.total_dim)
        self._quantizer_cache: dict = {}
        for band in (BAND_HIGH, BAND_NORMAL, BAND_LOW):
            kb, vb = BAND_COMPRESSION[band]
            self._quantizer_cache[("k", band, kb)] = LloydMaxQuantizer(n_bits=kb)
            self._quantizer_cache[("v", band, vb)] = LloydMaxQuantizer(n_bits=vb)
        self.k_quant_default = LloydMaxQuantizer(n_bits=k_bits)
        self.v_quant_default = LloydMaxQuantizer(n_bits=v_bits)
        self._quantizer_cache[("k", BAND_NORMAL, k_bits)] = self.k_quant_default
        self._quantizer_cache[("v", BAND_NORMAL, v_bits)] = self.v_quant_default
        self.entries: list[CacheEntry] = []
        self.holographic_entries: list[dict] = []
        self._global_step = 0
        self.hit_count = 0
        self.miss_count = 0
        self.refresh_count = 0
        self._attention_pattern: Optional[np.ndarray] = None
        self._allocation_bias: Optional[np.ndarray] = None

    @property
    def rotated_dim(self) -> int:
        return self._rotated_dim

    def _compute_spectral_entropy(self, vector: np.ndarray) -> float:
        return spectral_entropy(vector)

    def _compute_coherence(self, entry: CacheEntry) -> float:
        return landau_zener_coherence(
            self._global_step - entry.timestamp, half_life=1000.0
        )

    def _recency_fraction(self, entry: CacheEntry) -> float:
        if self._global_step == 0:
            return 1.0
        return max(0.0, 1.0 - (self._global_step - entry.timestamp) / self._global_step)

    def _frequency_fraction(self, entry: CacheEntry) -> float:
        mx = max((e.frequency for e in self.entries), default=1)
        return entry.frequency / mx

    def _cascade_eviction_score(self, entry: CacheEntry) -> float:
        return cascade_eviction_score(
            entropy=entry.entropy,
            coherence=self._compute_coherence(entry),
            recency=self._recency_fraction(entry),
            frequency=self._frequency_fraction(entry),
        )

    def _maybe_evict(self):
        if len(self.entries) < self.max_size:
            return
        scores = [self._cascade_eviction_score(e) for e in self.entries]
        self.entries.pop(int(np.argmin(scores)))

    def _adaptive_bits(self, position: int, default_bits: int) -> int:
        if self._allocation_bias is None or position >= len(self._allocation_bias):
            return default_bits
        extra = int(round(self._allocation_bias[position] * 4))
        return max(2, min(8, default_bits + extra))

    def _progressive_bits(self, default_bits: int, fill_ratio: float) -> int:
        if fill_ratio < 0.3:
            return self.progressive_start_bits
        elif fill_ratio < 0.6:
            return min(self.progressive_start_bits, default_bits + 2)
        return default_bits

    def _resonance_factor(self) -> float:
        if self.resonance_tracker is None:
            return 1.0
        rs = getattr(self.resonance_tracker, "resonance_score", None)
        if rs is not None:
            score = rs() if callable(rs) else rs
            return max(0.5, 1.0 - 0.5 * score)
        return 1.0

    def _assign_band(self, position: int, entropy: float) -> int:
        if position == 0 or entropy > 0.8:
            return BAND_HIGH
        if entropy < 0.3:
            return BAND_LOW
        return BAND_NORMAL

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        self._global_step += 1
        if self.enable_ar2:
            if self.ar2.should_skip_storage(key.ravel(), position):
                return
        if self.enable_holographic:
            k_enc = self.holographic.encode(key.ravel(), position)
            v_enc = self.holographic.encode(value.ravel(), position)
            self.holographic_entries.append(
                {
                    "k_enc": k_enc,
                    "v_enc": v_enc,
                    "position": position,
                    "timestamp": self._global_step,
                    "frequency": 1,
                    "coherence": 1.0,
                }
            )
            if self.enable_vlasov:
                self.vlasov.leapfrog_push(self.holographic_entries, self._global_step)
            if self.enable_mps and self.mps is not None:
                self.mps.entangle(key.ravel())
            return
        rotated_k = self.rotator.rotate(key.reshape(1, -1)).ravel()
        rotated_v = self.rotator.rotate(value.reshape(1, -1)).ravel()
        entropy = self._compute_spectral_entropy(key)
        band = self._assign_band(position, entropy)
        k_bits_band, v_bits_band = BAND_COMPRESSION[band]
        rf = self._resonance_factor()
        k_bits_band = max(2, int(k_bits_band * rf))
        v_bits_band = max(1, int(v_bits_band * rf))
        fill_ratio = len(self.entries) / max(self.max_size, 1)
        k_bits_prog = self._progressive_bits(k_bits_band, fill_ratio)
        v_bits_prog = self._progressive_bits(v_bits_band, fill_ratio)
        k_bits_final = self._adaptive_bits(position, k_bits_prog)
        v_bits_final = self._adaptive_bits(position, v_bits_prog)
        kq = self._get_quantizer("k", band, k_bits_final)
        vq = self._get_quantizer("v", band, v_bits_final)
        k_idx, _ = kq.compress(rotated_k)
        v_idx, _ = vq.compress(rotated_v)
        self._maybe_evict()
        self.entries.append(
            CacheEntry(
                k_indices=k_idx,
                v_indices=v_idx,
                position=position,
                band=band,
                entropy=entropy,
                coherence=1.0,
                timestamp=self._global_step,
                n_bits_k=k_bits_final,
                n_bits_v=v_bits_final,
                rotated_dim=self._rotated_dim,
            )
        )
        if (
            self.stochastic_refresh_interval > 0
            and self._global_step % self.stochastic_refresh_interval == 0
        ):
            self._stochastic_refresh()

    def store_batch(self, keys: np.ndarray, values: np.ndarray, positions: np.ndarray):
        for k, v, pos in zip(keys, values, positions):
            self.store(k, v, int(pos))

    def _get_quantizer(self, kind: str, band: int, n_bits: int):
        key = (kind, band, n_bits)
        if key not in self._quantizer_cache:
            self._quantizer_cache[key] = LloydMaxQuantizer(n_bits=n_bits)
        return self._quantizer_cache[key]

    def retrieve(self, position: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.enable_holographic:
            for entry in self.holographic_entries:
                if entry["position"] == position:
                    self.hit_count += 1
                    return (
                        self.holographic.decode(entry["k_enc"]).reshape(
                            self.n_heads, self.dim
                        ),
                        self.holographic.decode(entry["v_enc"]).reshape(
                            self.n_heads, self.dim
                        ),
                    )
        for entry in self.entries:
            if entry.position == position:
                band = entry.band
                kq = self._get_quantizer("k", band, entry.n_bits_k)
                vq = self._get_quantizer("v", band, entry.n_bits_v)
                k_recon = kq.decompress(entry.k_indices, (self._rotated_dim,))
                v_recon = vq.decompress(entry.v_indices, (self._rotated_dim,))
                entry.frequency += 1
                self.hit_count += 1
                return (
                    self.rotator.rotate(k_recon.reshape(1, -1)).ravel(),
                    self.rotator.rotate(v_recon.reshape(1, -1)).ravel(),
                )
        self.miss_count += 1
        return None

    def query(
        self, query_vector: np.ndarray, top_k: int = 10, use_compressed_sim: bool = True
    ) -> list[tuple[int, float, np.ndarray]]:
        q_rotated = self.rotator.rotate(query_vector.reshape(1, -1)).ravel()
        q_norm = np.linalg.norm(q_rotated) + 1e-10
        results = []
        for entry in self.entries:
            band = entry.band
            kq = self._get_quantizer("k", band, entry.n_bits_k)
            k_reconstructed = kq.decompress(entry.k_indices, (self._rotated_dim,))
            if use_compressed_sim:
                sim = float(np.dot(q_rotated, k_reconstructed)) / float(
                    q_norm * np.linalg.norm(k_reconstructed) + 1e-10
                )
            else:
                k_full = self.rotator.inverse_rotate(
                    k_reconstructed.reshape(1, -1)
                ).ravel()
                q_full = query_vector.ravel()
                sim = float(np.dot(q_full, k_full)) / float(
                    np.linalg.norm(q_full) * np.linalg.norm(k_full) + 1e-10
                )
            results.append((entry.position, sim, k_reconstructed))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def query_holographic(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float, np.ndarray]]:
        if not self.enable_holographic or not self.holographic_entries:
            return []
        q_flat = query_vector.ravel()
        q_norm = np.linalg.norm(q_flat) + 1e-10
        results = []
        for entry in self.holographic_entries:
            k_recon = self.holographic.decode(entry["k_enc"])
            sim = float(np.dot(q_flat, k_recon)) / float(
                q_norm * np.linalg.norm(k_recon) + 1e-10
            )
            results.append((entry["position"], sim, k_recon))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def update_attention_pattern(self, attention_scores: np.ndarray):
        key_weights = (
            np.mean(attention_scores, axis=0)
            if attention_scores.ndim == 2
            else attention_scores
        )
        self._attention_pattern = key_weights
        mn, mx = key_weights.min(), key_weights.max()
        if mx > mn:
            self._allocation_bias = (key_weights - mn) / (mx - mn)
        else:
            self._allocation_bias = np.ones_like(key_weights) * 0.5

    def predictive_evict(self, n_keep: int):
        if self._allocation_bias is None:
            return self._maybe_evict()
        scores = []
        for entry in self.entries:
            pos = entry.position
            bias = (
                self._allocation_bias[pos] if pos < len(self._allocation_bias) else 0.5
            )
            scores.append(self._cascade_eviction_score(entry) * (0.5 + bias))
        keep = set(np.argsort(scores)[-n_keep:])
        self.entries = [e for i, e in enumerate(self.entries) if i in keep]

    def _stochastic_refresh(self):
        if not self.entries:
            return
        idx = np.random.randint(0, len(self.entries))
        entry = self.entries[idx]
        kq = self._get_quantizer("k", entry.band, entry.n_bits_k)
        vq = self._get_quantizer("v", entry.band, entry.n_bits_v)
        k_recon = kq.decompress(entry.k_indices, (self._rotated_dim,))
        v_recon = vq.decompress(entry.v_indices, (self._rotated_dim,))
        k_idx_new, _ = kq.compress(self.rotator.rotate(k_recon.reshape(1, -1)).ravel())
        v_idx_new, _ = vq.compress(self.rotator.rotate(v_recon.reshape(1, -1)).ravel())
        entry.k_indices = k_idx_new
        entry.v_indices = v_idx_new
        entry.timestamp = self._global_step
        self.refresh_count += 1

    def clear(self):
        self.entries.clear()
        self.holographic_entries.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.refresh_count = 0
        self._global_step = 0
        self._attention_pattern = None
        self._allocation_bias = None

    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def compression_ratio(self) -> float:
        if not self.entries and not self.holographic_entries:
            return float(self.dim * 2)
        orig_bytes = 0
        comp_bytes = 0
        for _ in self.holographic_entries:
            orig_bytes += self.total_dim * 4 * 2
            comp_bytes += self.holographic.n_keep * self.holographic.quant_bits * 2 / 8
        for entry in self.entries:
            orig_bytes += self.dim * 2 * 2
            k_bytes = int(np.ceil(self._rotated_dim * entry.n_bits_k / 8))
            v_bytes = int(np.ceil(self._rotated_dim * entry.n_bits_v / 8))
            comp_bytes += k_bytes + v_bytes
        return float(orig_bytes / max(comp_bytes, 1))

    def num_positions(self) -> int:
        return len(self.entries) + len(self.holographic_entries)

    def cache_summary(self) -> dict:
        return {
            "num_positions": len(self.entries) + len(self.holographic_entries),
            "max_size": self.max_size,
            "hit_rate": self.hit_rate(),
            "compression_ratio": self.compression_ratio(),
            "refresh_count": self.refresh_count,
            "global_step": self._global_step,
            "use_dct": self.use_dct,
            "holographic_enabled": self.enable_holographic,
            "holographic_entries": len(self.holographic_entries),
            "holographic_n_keep": self.holographic.n_keep,
            "holographic_quant_bits": self.holographic.quant_bits,
        }

    def get_stats(self) -> dict:
        stats = self.cache_summary()
        n = len(self.holographic_entries)
        orig_per_token = self.total_dim * 4 * 2
        if self.enable_holographic and n > 0:
            n_keep = self.holographic.n_keep
            q_bits = self.holographic.quant_bits
            holographic_ratio = orig_per_token / max(n_keep * q_bits * 2 / 8.0, 1e-10)
            stats["holographic_ratio"] = round(holographic_ratio, 1)
            stats["theoretical_max_ratio"] = round(
                orig_per_token / (n_keep * 2 / 8.0), 1
            )
        else:
            stats["holographic_ratio"] = 1.0
            stats["theoretical_max_ratio"] = 1.0
        if self.enable_ar2:
            ar2_ratio = self.ar2.get_compression_ratio()
            stats["ar2_n_skipped"] = self.ar2._n_skipped
            stats["ar2_n_stored"] = self.ar2._n_stored
            stats["ar2_ratio"] = round(ar2_ratio, 2)
        else:
            stats["ar2_ratio"] = 1.0
        if self.enable_vlasov:
            stats["vlasov_step"] = self.vlasov._step
            stats["vlasov_entries"] = len(self.holographic_entries)
        if self.enable_multiband:
            mb_bits = self.multiband.estimate_total_bits(self._global_step)
            mb_ratio = self._global_step * self.total_dim * 4 * 2 * 8 / max(mb_bits, 1)
            stats["multiband_estimated_ratio"] = round(mb_ratio, 1)
        else:
            stats["multiband_estimated_ratio"] = 1.0
        if self.enable_hybrid_quant:
            stats["hybrid_quant_trained"] = self.hybrid_quant.trained
            stats["hybrid_quant_ratio"] = round(
                self.hybrid_quant.compression_ratio(), 1
            )
        if self.enable_mps and self.mps is not None:
            stats["mps_n_entries"] = self.mps.n_entries
            stats["mps_bond_dim"] = self.mps.bond_dim
            stats["mps_compression_ratio"] = round(self.mps.compression_ratio(), 1)
        effective = stats.get("holographic_ratio", 1.0) * stats.get("ar2_ratio", 1.0)
        stats["effective_compression"] = round(effective, 1)
        n_keep = self.holographic.n_keep
        q_bits = self.holographic.quant_bits
        stats["original_bytes"] = int(n * self.total_dim * 4 * 2)
        stats["compressed_bytes"] = round(n * n_keep * q_bits * 2 / 8.0, 1)
        stats["n_entries"] = n
        return stats


def _pack_int2(values: np.ndarray) -> np.ndarray:
    n = len(values)
    clamped = np.clip(np.round(values).astype(np.int8), -2, 1) & 0x03
    packed = np.zeros((n + 3) // 4, dtype=np.uint8)
    for i in range(0, n, 4):
        b = 0
        for j in range(4):
            if i + j < n:
                b |= (int(clamped[i + j]) & 0x03) << (j * 2)
        packed[i // 4] = b
    return packed


def _unpack_int2(packed: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        byte_val = int(packed[i // 4])
        shift = (i % 4) * 2
        val = (byte_val >> shift) & 0x03
        if val >= 2:
            val -= 4
        result[i] = np.int8(val)
    return result


def _pack_int4(values: np.ndarray) -> np.ndarray:
    n = len(values)
    clamped = np.clip(np.round(values).astype(np.int8), -8, 7) & 0x0F
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    for i in range(0, n, 2):
        lo = int(clamped[i]) & 0x0F
        hi = int(clamped[i + 1]) & 0x0F if i + 1 < n else 0
        packed[i // 2] = (hi << 4) | lo
    return packed


def _unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        byte_val = int(packed[i // 2])
        if i % 2 == 0:
            val = byte_val & 0x0F
        else:
            val = (byte_val >> 4) & 0x0F
        if val >= 8:
            val -= 16
        result[i] = np.int8(val)
    return result


class ResidualCodec:
    def __init__(self):
        self._min_val: Optional[int] = None
        self._max_val: Optional[int] = None

    def encode(self, data: np.ndarray) -> bytes:
        flat = data.ravel().astype(np.int32)
        self._min_val = int(np.min(flat))
        self._max_val = int(np.max(flat))
        shifted = flat - self._min_val
        if self._max_val - self._min_val < 256:
            packed = shifted.astype(np.uint8).tobytes()
        else:
            packed = shifted.astype(np.uint16).tobytes()
        compressed = zlib.compress(packed, level=3)
        header = struct.pack("<iii", self._min_val, self._max_val, len(flat))
        return header + compressed

    def decode(self, encoded: bytes, n: int) -> np.ndarray:
        self._min_val = struct.unpack("<i", encoded[0:4])[0]
        self._max_val = struct.unpack("<i", encoded[4:8])[0]
        compressed = encoded[12:]
        packed = zlib.decompress(compressed)
        dtype = np.uint8 if self._max_val - self._min_val < 256 else np.uint16
        shifted = np.frombuffer(packed, dtype=dtype).astype(np.int32)
        result = shifted + self._min_val
        if len(result) < n:
            result = np.pad(result, (0, n - len(result)), "constant")
        return result[:n]


class FreqKVExtremeCompressor:
    def __init__(
        self,
        head_dim: int = 128,
        target_ratio: float = 384.0,
        dct_keep_ratio: float = 0.02,
        key_group_size: int = 64,
        val_group_size: int = 128,
        key_bits: int = 4,
        val_bits: int = 4,
        gear_rank: int = 4,
        gear_sparse_frac: float = 0.001,
    ):
        self.head_dim = head_dim
        self.target_ratio = target_ratio
        self.dct_keep_ratio = dct_keep_ratio
        self.key_group_size = key_group_size
        self.val_group_size = val_group_size
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.gear_rank = gear_rank
        self.gear_sparse_frac = gear_sparse_frac
        self._lock = threading.Lock()

    def _compute_cumulative_energy(self, dct_coeffs: np.ndarray) -> np.ndarray:
        energy = dct_coeffs**2
        return np.cumsum(energy, axis=0) / (np.sum(energy, axis=0, keepdims=True) + EPS)

    def _kivi_quantize_keys(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq_len, dim = keys.shape
        group_size = min(self.key_group_size, dim)
        n_groups = max(1, dim // group_size)
        q_indices = np.zeros((seq_len, dim), dtype=np.int8)
        scales = np.zeros((seq_len, n_groups), dtype=np.float32)
        half = 1 << (self.key_bits - 1)
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, dim)
            group_data = keys[:, start:end]
            amax = np.max(np.abs(group_data), axis=1, keepdims=True)
            amax = np.clip(amax, EPS, None)
            scaled = np.clip(
                np.round(group_data / amax * (half - 1)), -(half - 1), half - 1
            )
            q_indices[:, start:end] = scaled.astype(np.int8)
            scales[:, g] = amax.ravel()
        return q_indices, scales

    def _kivi_quantize_values(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len, dim = values.shape
        group_size = min(self.val_group_size, dim)
        n_groups = max(1, dim // group_size)
        q_indices = np.zeros((seq_len, dim), dtype=np.int8)
        scales = np.zeros((seq_len, n_groups), dtype=np.float32)
        half = 1 << (self.val_bits - 1)
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, dim)
            group_data = values[:, start:end]
            amax = np.max(np.abs(group_data), axis=1, keepdims=True)
            amax = np.clip(amax, EPS, None)
            scaled = np.clip(
                np.round(group_data / amax * (half - 1)), -(half - 1), half - 1
            )
            q_indices[:, start:end] = scaled.astype(np.int8)
            scales[:, g] = amax.ravel()
        return q_indices, scales

    def _gear_error_recovery(
        self, error: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = min(self.gear_rank, min(error.shape) - 1)
        if r < 1:
            return error.copy(), np.zeros((0,)), np.zeros((0,))
        U, S, Vt = np.linalg.svd(error, full_matrices=False)
        low_rank = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        sparse_residual = error - low_rank
        n_total = sparse_residual.size
        n_sparse = max(1, int(n_total * self.gear_sparse_frac))
        flat = sparse_residual.ravel()
        top_idx = np.argpartition(np.abs(flat), -n_sparse)[-n_sparse:]
        sparse_fix = np.zeros_like(flat)
        sparse_fix[top_idx] = flat[top_idx]
        return low_rank + sparse_fix.reshape(error.shape), U[:, :r], Vt[:r, :]

    def compress(self, seq: np.ndarray) -> dict:
        seq = np.ascontiguousarray(seq, dtype=np.float32)
        orig_seq_len, head_dim = seq.shape
        dct_coeffs = dct(seq.astype(np.float64), axis=0)
        n_keep = max(1, int(orig_seq_len * self.dct_keep_ratio))
        n_keep = min(n_keep, orig_seq_len)
        cum_energy = self._compute_cumulative_energy(dct_coeffs)
        min_k = max(1, orig_seq_len // 100)
        n_keep = max(min_k, n_keep)
        for k in range(1, orig_seq_len + 1):
            if np.mean(cum_energy[k - 1, :]) >= 0.99:
                n_keep = max(n_keep, k)
                break
        n_keep = min(n_keep, orig_seq_len)
        dct_truncated = dct_coeffs[:n_keep, :].copy()
        if dct_truncated.shape[0] < 2:
            dct_padded = np.zeros((orig_seq_len, head_dim), dtype=np.float64)
            dct_padded[:n_keep] = dct_truncated
            return {
                "shape": (orig_seq_len, head_dim),
                "n_keep": n_keep,
                "dct_coeffs": dct_truncated.astype(np.float16).tobytes(),
                "dct_coeffs_shape": dct_truncated.shape,
                "key_scales": np.array([1.0], dtype=np.float16).tobytes(),
                "val_scales": np.array([1.0], dtype=np.float16).tobytes(),
                "key_q": np.zeros(1, dtype=np.uint8).tobytes(),
                "val_q": np.zeros(1, dtype=np.uint8).tobytes(),
                "gear_components": (
                    np.zeros(1, dtype=np.uint8).tobytes(),
                    np.zeros(1, dtype=np.uint8).tobytes(),
                ),
                "compressed_bytes": 64,
            }
        keys_part = dct_truncated[:, : head_dim // 2].copy()
        vals_part = (
            dct_truncated[:, head_dim // 2 :].copy()
            if head_dim > 1
            else dct_truncated[:, :1].copy()
        )
        k_indices, k_scales = self._kivi_quantize_keys(keys_part)
        v_indices, v_scales = self._kivi_quantize_values(vals_part)
        half_k = 1 << (self.key_bits - 1)
        half_v = 1 << (self.val_bits - 1)
        k_deq = np.zeros_like(keys_part, dtype=np.float64)
        for g in range(k_scales.shape[1]):
            start = g * self.key_group_size
            end = min(start + self.key_group_size, keys_part.shape[1])
            for t in range(k_scales.shape[0]):
                amax = k_scales[t, g]
                k_deq[t, start:end] = (
                    k_indices[t, start:end].astype(np.float64) * amax / max(half_k, 1)
                )
        v_deq = np.zeros_like(vals_part, dtype=np.float64)
        for g in range(v_scales.shape[1]):
            start = g * self.val_group_size
            end = min(start + self.val_group_size, vals_part.shape[1])
            for t in range(v_scales.shape[0]):
                amax = v_scales[t, g]
                v_deq[t, start:end] = (
                    v_indices[t, start:end].astype(np.float64) * amax / max(half_v, 1)
                )
        k_recovered, kU, kVt = self._gear_error_recovery(
            keys_part.astype(np.float64) - k_deq
        )
        v_recovered, vU, vVt = self._gear_error_recovery(
            vals_part.astype(np.float64) - v_deq
        )
        k_indices_packed = _pack_int4(k_indices.ravel().astype(np.float32))
        v_indices_packed = _pack_int4(v_indices.ravel().astype(np.float32))
        return {
            "shape": (orig_seq_len, head_dim),
            "n_keep": n_keep,
            "key_indices": k_indices_packed.tobytes(),
            "val_indices": v_indices_packed.tobytes(),
            "key_indices_shape": k_indices.shape,
            "val_indices_shape": v_indices.shape,
            "key_scales": k_scales.astype(np.float16).tobytes(),
            "val_scales": v_scales.astype(np.float16).tobytes(),
            "key_scales_shape": k_scales.shape,
            "val_scales_shape": v_scales.shape,
            "gear_kU": kU.astype(np.float16).tobytes() if kU.size > 0 else b"",
            "gear_kVt": kVt.astype(np.float16).tobytes() if kVt.size > 0 else b"",
            "gear_vU": vU.astype(np.float16).tobytes() if vU.size > 0 else b"",
            "gear_vVt": vVt.astype(np.float16).tobytes() if vVt.size > 0 else b"",
            "compressed_bytes": len(k_indices_packed) + len(v_indices_packed) + 1024,
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        orig_seq_len, head_dim = compressed["shape"]
        n_keep = compressed["n_keep"]
        k_shape = compressed.get(
            "key_indices_shape", (max(1, n_keep), max(1, head_dim // 2))
        )
        v_shape = compressed.get(
            "val_indices_shape", (max(1, n_keep), max(1, head_dim - head_dim // 2))
        )
        n_k_total = k_shape[0] * k_shape[1]
        n_v_total = v_shape[0] * v_shape[1]
        k_indices = (
            _unpack_int4(
                np.frombuffer(compressed["key_indices"], dtype=np.uint8).copy(),
                n_k_total,
            )
            .astype(np.float32)
            .reshape(k_shape)
        )
        v_indices = (
            _unpack_int4(
                np.frombuffer(compressed["val_indices"], dtype=np.uint8).copy(),
                n_v_total,
            )
            .astype(np.float32)
            .reshape(v_shape)
        )
        k_scales = (
            np.frombuffer(compressed["key_scales"], dtype=np.float16)
            .copy()
            .astype(np.float32)
        )
        v_scales = (
            np.frombuffer(compressed["val_scales"], dtype=np.float16)
            .copy()
            .astype(np.float32)
        )
        try:
            k_scales = k_scales.reshape(
                compressed.get(
                    "key_scales_shape",
                    (k_shape[0], max(1, k_shape[1] // self.key_group_size)),
                )
            )
        except ValueError:
            k_scales = np.ones((k_shape[0], 1), dtype=np.float32)
        try:
            v_scales = v_scales.reshape(
                compressed.get(
                    "val_scales_shape",
                    (v_shape[0], max(1, v_shape[1] // self.val_group_size)),
                )
            )
        except ValueError:
            v_scales = np.ones((v_shape[0], 1), dtype=np.float32)
        half_k = 1 << (self.key_bits - 1)
        half_v = 1 << (self.val_bits - 1)
        k_deq = np.zeros(k_shape, dtype=np.float64)
        for g in range(k_scales.shape[1]):
            start = g * self.key_group_size
            end = min(start + self.key_group_size, k_shape[1])
            for t in range(k_shape[0]):
                amax = (
                    k_scales[t, g]
                    if k_scales.ndim > 1
                    else k_scales[min(g, k_scales.shape[0] - 1)]
                )
                k_deq[t, start:end] = (
                    k_indices[t, start:end].astype(np.float64) * amax / max(half_k, 1)
                )
        v_deq = np.zeros(v_shape, dtype=np.float64)
        for g in range(v_scales.shape[1]):
            start = g * self.val_group_size
            end = min(start + self.val_group_size, v_shape[1])
            for t in range(v_shape[0]):
                amax = (
                    v_scales[t, g]
                    if v_scales.ndim > 1
                    else v_scales[min(g, v_scales.shape[0] - 1)]
                )
                v_deq[t, start:end] = (
                    v_indices[t, start:end].astype(np.float64) * amax / max(half_v, 1)
                )
        try:
            gear_kU = (
                np.frombuffer(compressed.get("gear_kU", b""), dtype=np.float16)
                .copy()
                .astype(np.float64)
            )
            gear_kVt = (
                np.frombuffer(compressed.get("gear_kVt", b""), dtype=np.float16)
                .copy()
                .astype(np.float64)
            )
            gear_vU = (
                np.frombuffer(compressed.get("gear_vU", b""), dtype=np.float16)
                .copy()
                .astype(np.float64)
            )
            gear_vVt = (
                np.frombuffer(compressed.get("gear_vVt", b""), dtype=np.float16)
                .copy()
                .astype(np.float64)
            )
            if gear_kU.size > 0 and gear_kVt.size > 0:
                try:
                    k_deq = k_deq + gear_kU.reshape(
                        k_shape[0], self.gear_rank
                    ) @ gear_kVt.reshape(self.gear_rank, k_shape[1])
                except (ValueError, np.linalg.LinAlgError):
                    pass
            if gear_vU.size > 0 and gear_vVt.size > 0:
                try:
                    v_deq = v_deq + gear_vU.reshape(
                        v_shape[0], self.gear_rank
                    ) @ gear_vVt.reshape(self.gear_rank, v_shape[1])
                except (ValueError, np.linalg.LinAlgError):
                    pass
        except Exception:
            pass
        dct_recon = np.zeros((orig_seq_len, head_dim), dtype=np.float64)
        if k_shape[1] + v_shape[1] >= head_dim:
            dct_recon[:n_keep, : k_shape[1]] = k_deq.astype(np.float64)
            dct_recon[:n_keep, k_shape[1] : k_shape[1] + v_shape[1]] = v_deq.astype(
                np.float64
            )
        else:
            concat = np.concatenate([k_deq, v_deq], axis=1)
            dct_recon[:n_keep, : min(head_dim, concat.shape[1])] = concat[
                :, : min(head_dim, concat.shape[1])
            ]
        return idct(dct_recon, axis=0).astype(np.float32)


class PlasmaConfinementKVEvictor:
    def __init__(
        self,
        dim: int = 128,
        n_particles: int = 4096,
        confinement_threshold: float = 0.1,
        leapfrog_dt: float = 0.01,
        charge_decay: float = 0.99,
        field_width: float = 1.0,
    ):
        self.dim = dim
        self.n_particles = n_particles
        self.confinement_threshold = confinement_threshold
        self.leapfrog_dt = leapfrog_dt
        self.charge_decay = charge_decay
        self.field_width = field_width
        self._lock = threading.Lock()
        self._positions: list[int] = []
        self._charges: list[float] = []
        self._velocities: list[float] = []
        self._potential_energies: list[float] = []
        self._timestep = 0
        self._evicted_count = 0
        self._confined_count = 0

    def compute_charge(
        self, attention_weight: float, recency_weight: float, position: int
    ) -> float:
        return attention_weight + recency_weight

    def register_particle(self, position: int, charge: float):
        with self._lock:
            self._positions.append(position)
            self._charges.append(charge)
            self._velocities.append(0.0)
            self._potential_energies.append(0.0)
            if len(self._positions) > self.n_particles:
                self._evict_one()

    def _poisson_potential(self, pos_idx: int) -> float:
        n = len(self._positions)
        if n < 2:
            return 0.0
        potential = 0.0
        xi = self._positions[pos_idx]
        for j, (xj, qj) in enumerate(zip(self._positions, self._charges)):
            if j == pos_idx:
                continue
            dx = xi - xj
            potential += qj * np.exp(-0.5 * (dx / self.field_width) ** 2)
        return float(potential)

    def _leapfrog_step(self):
        n = len(self._positions)
        if n < 2:
            return
        half_dt = self.leapfrog_dt * 0.5
        pos_arr = np.array(self._positions, dtype=np.float64)
        chg_arr = np.array(self._charges, dtype=np.float64)
        for i in range(n):
            dx = pos_arr - pos_arr[i]
            f = (
                chg_arr
                * dx
                * np.exp(-0.5 * (dx / self.field_width) ** 2)
                / (self.field_width**2)
            )
            f[i] = 0.0
            self._velocities[i] += float(np.sum(f)) * half_dt
        for i in range(n):
            self._positions[i] = max(
                0,
                int(round(self._positions[i] + self._velocities[i] * self.leapfrog_dt)),
            )
        for i in range(n):
            dx = pos_arr - pos_arr[i]
            f = (
                chg_arr
                * dx
                * np.exp(-0.5 * (dx / self.field_width) ** 2)
                / (self.field_width**2)
            )
            f[i] = 0.0
            self._velocities[i] += float(np.sum(f)) * half_dt
            self._velocities[i] *= 0.999

    def update_charges(self, attention_map: dict[int, float]):
        with self._lock:
            for i in range(len(self._charges)):
                pos = self._positions[i]
                attn = attention_map.get(pos, 0.0)
                recency = 1.0 / (1.0 + self._timestep)
                self._charges[i] = (
                    self.compute_charge(attn, recency, pos) * self.charge_decay
                )
            self._timestep += 1

    def _evict_one(self) -> Optional[int]:
        n = len(self._positions)
        if n == 0:
            return None
        for i in range(n):
            self._potential_energies[i] = self._poisson_potential(i)
        min_idx = int(np.argmin(self._potential_energies))
        if self._potential_energies[min_idx] >= self.confinement_threshold:
            return None
        evicted_pos = self._positions.pop(min_idx)
        self._charges.pop(min_idx)
        self._velocities.pop(min_idx)
        self._potential_energies.pop(min_idx)
        self._evicted_count += 1
        return evicted_pos

    def get_eviction_candidates(self, n: int = 1) -> list[int]:
        with self._lock:
            self._leapfrog_step()
            candidates = []
            for _ in range(n):
                pos = self._evict_one()
                if pos is not None:
                    candidates.append(pos)
                else:
                    break
            return candidates

    def record_access(self, position: int, attention_weight: float = 0.0):
        with self._lock:
            if position in self._positions:
                idx = self._positions.index(position)
                self._charges[idx] += attention_weight
                self._confined_count += 1

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "n_particles": len(self._positions),
                "max_particles": self.n_particles,
                "evicted_count": self._evicted_count,
                "confined_count": self._confined_count,
                "timestep": self._timestep,
                "mean_charge": float(np.mean(self._charges)) if self._charges else 0.0,
                "mean_potential": float(np.mean(self._potential_energies))
                if self._potential_energies
                else 0.0,
                "threshold": self.confinement_threshold,
            }


class PredictiveCodingKVCache_v2:
    def __init__(
        self,
        dim: int = 128,
        max_order: int = 4,
        significance_level: float = 0.05,
        residual_bits: int = 8,
        min_sequence_len: int = 8,
    ):
        self.dim = dim
        self.max_order = max_order
        self.significance_level = significance_level
        self.residual_bits = residual_bits
        self.min_sequence_len = min_sequence_len
        self._lock = threading.Lock()
        self._history: list[np.ndarray] = []
        self._residuals: dict[int, np.ndarray] = {}
        self._ar_coeffs: Optional[np.ndarray] = None
        self._ar_order: int = 0
        self._warmup: list[np.ndarray] = []
        self._rng = np.random.RandomState(0)
        self._n_skipped = 0
        self._n_stored = 0
        self._n_decoded = 0
        self._total_original_energy = 0.0
        self._total_residual_energy = 0.0

    def _autocorr(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        xd = x.astype(np.float64) - np.mean(x.astype(np.float64))
        n = len(xd)
        r = np.correlate(xd, xd, mode="full")[n - 1 : n + max_lag]
        return r / max(n, 1)

    def _yule_walker(self, x: np.ndarray, order: int) -> np.ndarray:
        n = len(x)
        if n <= order + 2:
            return np.zeros(order, dtype=np.float64)
        r = self._autocorr(x, order + 1)
        try:
            R = np.zeros((order, order), dtype=np.float64)
            for i in range(order):
                for j in range(order):
                    R[i, j] = r[abs(i - j)]
            coeffs = np.linalg.solve(R, r[1 : order + 1])
            spec_rad = np.max(np.abs(np.roots(np.r_[1, -coeffs])))
            if spec_rad >= 0.99:
                coeffs = coeffs * (0.99 / max(spec_rad, 1e-10))
            return coeffs
        except np.linalg.LinAlgError:
            return np.zeros(order, dtype=np.float64)

    def _compute_pacf(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        n = len(x)
        if n < max_lag + 2:
            return np.zeros(max_lag)
        pacf = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            if lag == 1:
                r = self._autocorr(x, 2)
                pacf[0] = r[1] / max(r[0], EPS)
            else:
                coeffs = self._yule_walker(x, lag)
                pacf[lag - 1] = coeffs[-1] if len(coeffs) >= lag else 0.0
        return pacf

    def _select_order(self, x: np.ndarray) -> int:
        if len(x) < self.min_sequence_len:
            return 0
        max_order = min(self.max_order, len(x) - 3)
        if max_order < 1:
            return 0
        pacf = self._compute_pacf(x, max_order)
        threshold = 2.0 / np.sqrt(max(len(x), 2))
        significant_lags = np.where(np.abs(pacf) > threshold)[0]
        if len(significant_lags) == 0:
            return 0
        return min(int(significant_lags[-1]) + 1, max_order)

    def _residual_quantize(
        self, residual: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        amax = float(np.max(np.abs(residual)))
        if amax < EPS:
            return np.zeros_like(residual, dtype=np.int32), 0.0, 0.0
        half = 1 << (self.residual_bits - 1)
        q = np.clip(np.round(residual / amax * (half - 1)), -(half - 1), half - 1)
        return q.astype(np.int32), amax, float(np.mean(q**2))

    def _residual_dequantize(self, q: np.ndarray, amax: float) -> np.ndarray:
        if amax < EPS:
            return np.zeros_like(q, dtype=np.float64)
        half = 1 << (self.residual_bits - 1)
        return q.astype(np.float64) * (amax / max(half - 1, 1))

    def encode_sequence(self, seq: np.ndarray) -> dict:
        seq = np.ascontiguousarray(seq, dtype=np.float32)
        orig_shape = seq.shape
        is_flat = seq.ndim == 1
        if is_flat:
            seq = seq.reshape(-1, 1)
        orig_len, dim = seq.shape
        with self._lock:
            flat_sample = seq[:, 0].copy() if dim > 0 else np.zeros(orig_len)
            order = self._select_order(flat_sample)
            if order < 1:
                return {
                    "type": "raw",
                    "shape": orig_shape,
                    "data": seq.tobytes(),
                    "compressed_bytes": seq.nbytes,
                    "ar_order": 0,
                }
            coeffs = self._yule_walker(flat_sample, order)
            if np.max(np.abs(coeffs)) < 1e-10:
                return {
                    "type": "raw",
                    "shape": orig_shape,
                    "data": seq.tobytes(),
                    "compressed_bytes": seq.nbytes,
                    "ar_order": 0,
                }
            residuals_by_chan = []
            warmup_by_chan = []
            total_resid_energy = 0.0
            total_orig_energy = 0.0
            for c in range(dim):
                x = seq[:, c].astype(np.float64)
                resid = np.zeros(orig_len, dtype=np.float64)
                warm = x[:order].copy()
                for i in range(order, orig_len):
                    resid[i] = x[i] - np.dot(coeffs, x[i - order : i][::-1])
                residuals_by_chan.append(resid[order:])
                warmup_by_chan.append(warm)
                total_resid_energy += float(np.mean(resid[order:] ** 2))
                total_orig_energy += float(np.mean(x**2))
            q_resid, amax, _ = self._residual_quantize(
                np.concatenate(residuals_by_chan)
            )
            codec = ResidualCodec()
            encoded = codec.encode(q_resid)
            self._ar_coeffs = coeffs
            self._ar_order = order
            self._warmup = warmup_by_chan
            self._total_original_energy += total_orig_energy + EPS
            self._total_residual_energy += total_resid_energy + EPS
            return {
                "type": "ar_coded",
                "shape": orig_shape,
                "ar_order": order,
                "dim": dim,
                "ar_coeffs": coeffs.astype(np.float32).tobytes(),
                "warmup": np.array(warmup_by_chan, dtype=np.float32).tobytes(),
                "warmup_shape": (order, dim),
                "residuals_encoded": encoded,
                "residual_amax": np.float32(amax).tobytes(),
                "compressed_bytes": len(encoded) + order * dim * 4 + 32,
            }

    def decode_sequence(self, encoded: dict) -> np.ndarray:
        if encoded.get("type") == "raw":
            return np.frombuffer(encoded["data"], dtype=np.float32).reshape(
                encoded["shape"]
            )
        shape = encoded["shape"]
        order = encoded["ar_order"]
        if order < 1:
            return np.zeros(shape, dtype=np.float32)
        is_flat = len(shape) == 1
        seq_len = shape[0]
        dim = encoded.get("dim", 1 if is_flat else shape[1])
        coeffs = (
            np.frombuffer(encoded["ar_coeffs"], dtype=np.float32)
            .copy()
            .astype(np.float64)
        )
        warmup_raw = (
            np.frombuffer(encoded["warmup"], dtype=np.float32).copy().astype(np.float64)
        )
        warmup_shape = encoded.get("warmup_shape", (order, dim))
        try:
            warmup = warmup_raw.reshape(warmup_shape)
        except ValueError:
            warmup = np.zeros((order, dim), dtype=np.float64)
        amax = float(np.frombuffer(encoded["residual_amax"], dtype=np.float32)[0])
        n_resid_per_chan = max(0, seq_len - order)
        n_total_resid = n_resid_per_chan * dim
        if n_total_resid <= 0:
            return np.zeros(shape, dtype=np.float32)
        try:
            codec = ResidualCodec()
            q_resid = codec.decode(encoded["residuals_encoded"], n_total_resid)
        except Exception:
            return np.zeros(shape, dtype=np.float32)
        resid = self._residual_dequantize(q_resid, amax)
        result = np.zeros((seq_len, dim), dtype=np.float64)
        for c in range(dim):
            if order > 0:
                if warmup.ndim > 1 and warmup.shape[0] >= order and warmup.shape[1] > c:
                    result[:order, c] = warmup[:order, c]
                elif warmup.ndim == 1:
                    result[: min(order, len(warmup)), c] = warmup[
                        : min(order, len(warmup))
                    ]
            for i in range(n_resid_per_chan):
                idx = order + i
                r_idx = i * dim + c
                result[idx, c] = np.dot(coeffs, result[idx - order : idx, c][::-1]) + (
                    resid[r_idx] if r_idx < len(resid) else 0.0
                )
        with self._lock:
            self._n_decoded += 1
        return result.astype(np.float32).reshape(shape)

    def get_compression_ratio_estimate(self) -> float:
        with self._lock:
            if self._total_original_energy < EPS:
                return 1.0
            return float(
                self._total_original_energy / max(self._total_residual_energy, EPS)
            )

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "ar_order": self._ar_order,
                "n_skipped": self._n_skipped,
                "n_stored": self._n_stored,
                "n_decoded": self._n_decoded,
                "compression_ratio": self.get_compression_ratio_estimate(),
                "total_orig_energy": self._total_original_energy,
                "total_resid_energy": self._total_residual_energy,
            }


class SpectralSuperpositionKVCache:
    def __init__(
        self,
        head_dim: int = 128,
        max_entries: int = 65536,
        theta: float = 1.0,
        retrieval_iterations: int = 3,
    ):
        self.head_dim = head_dim
        self.max_entries = max_entries
        self.theta = theta
        self.retrieval_iterations = retrieval_iterations
        self._lock = threading.Lock()
        self._superposed_k: Optional[np.ndarray] = None
        self._superposed_v: Optional[np.ndarray] = None
        self._entry_count = 0
        self._entry_positions: list[int] = []
        self._phase_keys: dict[int, np.ndarray] = {}
        self._retrieval_hits = 0
        self._retrieval_misses = 0

    def _dct_encode(self, vector: np.ndarray) -> np.ndarray:
        c = dct(vector.ravel().astype(np.float64))
        angle = self.theta * c
        return np.cos(angle) + 1j * np.sin(angle)

    def _dct_decode(self, phase: np.ndarray) -> np.ndarray:
        return idct(np.angle(phase) / max(self.theta, EPS))

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        with self._lock:
            k_phase = self._dct_encode(key.ravel())
            v_phase = self._dct_encode(value.ravel())
            if self._superposed_k is None:
                self._superposed_k = k_phase.copy().astype(np.complex128)
                self._superposed_v = v_phase.copy().astype(np.complex128)
            else:
                self._superposed_k += k_phase
                self._superposed_v += v_phase
            self._entry_positions.append(position)
            self._phase_keys[position] = k_phase.copy()
            self._entry_count += 1
            if self._entry_count > self.max_entries:
                self._prune_oldest()

    def _prune_oldest(self):
        if not self._entry_positions:
            return
        oldest = self._entry_positions.pop(0)
        if oldest in self._phase_keys:
            old_phase = self._phase_keys.pop(oldest)
            if self._superposed_k is not None:
                self._superposed_k -= old_phase
            self._entry_count -= 1

    def retrieve(self, position: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if (
                self._superposed_k is None
                or self._entry_count == 0
                or position not in self._phase_keys
            ):
                self._retrieval_misses += 1
                return None
            query_k = self._phase_keys[position]
            conj_k = np.conj(query_k)
            conj_v = conj_k.copy()
            for entry_pos in reversed(self._entry_positions):
                if entry_pos == position:
                    ep = self._phase_keys.get(entry_pos)
                    if ep is not None:
                        conj_v = np.conj(ep)
                    break
            result_k = self._superposed_k * conj_k
            result_v = self._superposed_v * conj_v
            for _ in range(self.retrieval_iterations - 1):
                interference_k = np.zeros_like(result_k)
                for other_pos in self._entry_positions:
                    if other_pos == position:
                        continue
                    other_k = self._phase_keys.get(other_pos)
                    if other_k is not None:
                        interference_k += (
                            other_k
                            * np.dot(result_k, np.conj(other_k))
                            / max(self.head_dim, 1)
                        )
                result_k = self._superposed_k * conj_k - interference_k * 0.1
                result_v = self._superposed_v * conj_v
            self._retrieval_hits += 1
            return (
                self._dct_decode(result_k).astype(np.float32).reshape(-1),
                self._dct_decode(result_v).astype(np.float32).reshape(-1),
            )

    def query(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        with self._lock:
            q_phase = self._dct_encode(query_vector.ravel())
            scores = []
            for pos in self._entry_positions:
                entry_phase = self._phase_keys.get(pos)
                if entry_phase is None:
                    continue
                overlap = float(
                    np.abs(np.dot(q_phase.ravel(), np.conj(entry_phase.ravel())))
                )
                overlap /= max(
                    np.linalg.norm(q_phase) * np.linalg.norm(entry_phase), EPS
                )
                scores.append((pos, overlap))
            scores.sort(key=lambda x: -x[1])
            return scores[:top_k]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "entry_count": self._entry_count,
                "max_entries": self.max_entries,
                "retrieval_hits": self._retrieval_hits,
                "retrieval_misses": self._retrieval_misses,
                "head_dim": self.head_dim,
                "superposed_k_norm": float(np.linalg.norm(self._superposed_k))
                if self._superposed_k is not None
                else 0.0,
            }


class TimeCrystalKVCache_v2:
    def __init__(
        self,
        head_dim: int = 128,
        max_entries: int = 16384,
        pump_interval: int = 1024,
        phase_bits: int = 4,
        compression_ratio: float = 100.0,
    ):
        self.head_dim = head_dim
        self.max_entries = max_entries
        self.pump_interval = pump_interval
        self.phase_bits = phase_bits
        self.compression_ratio = compression_ratio
        self._lock = threading.Lock()
        self._phase0_k: dict[int, tuple] = {}
        self._phase0_v: dict[int, tuple] = {}
        self._phase1_k: dict[int, tuple] = {}
        self._phase1_v: dict[int, tuple] = {}
        self._fp32_k: dict[int, np.ndarray] = {}
        self._fp32_v: dict[int, np.ndarray] = {}
        self._entry_positions: list[int] = []
        self._global_step = 0
        self._pump_count = 0
        self._retrieval_count = 0
        self._error_estimate = 0.0

    def _quantize_phase(self, vector: np.ndarray, phase: int) -> tuple:
        half = 1 << (self.phase_bits - 1)
        amax = float(np.max(np.abs(vector))) + EPS
        scaled = vector / amax * (half - 1)
        if phase == 1:
            scaled = -scaled
        q = np.clip(np.round(scaled), -(half - 1), half - 1)
        return q.astype(np.int8), amax

    def _dequantize_phase(self, q: np.ndarray, amax: float, phase: int) -> np.ndarray:
        half = 1 << (self.phase_bits - 1)
        dq = q.astype(np.float64) * (amax / max(half - 1, 1))
        return dq if phase == 0 else -dq

    def store(self, key: np.ndarray, value: np.ndarray, position: int):
        with self._lock:
            k_flat = key.ravel().astype(np.float64)
            v_flat = value.ravel().astype(np.float64)
            dct_k = dct(k_flat)
            dct_v = dct(v_flat)
            self._phase0_k[position] = self._quantize_phase(dct_k, 0)
            self._phase0_v[position] = self._quantize_phase(dct_v, 0)
            self._phase1_k[position] = self._quantize_phase(dct_k, 1)
            self._phase1_v[position] = self._quantize_phase(dct_v, 1)
            self._fp32_k[position] = k_flat.astype(np.float32)
            self._fp32_v[position] = v_flat.astype(np.float32)
            self._entry_positions.append(position)
            self._global_step += 1
            if self._global_step % self.pump_interval == 0:
                self._pump()
            if len(self._entry_positions) > self.max_entries:
                self._evict_one()

    def _pump(self):
        for pos in self._entry_positions:
            if pos in self._fp32_k:
                k_orig = self._fp32_k[pos].astype(np.float64)
                v_orig = self._fp32_v[pos].astype(np.float64)
                dct_k = dct(k_orig)
                dct_v = dct(v_orig)
                self._phase0_k[pos] = self._quantize_phase(dct_k, 0)
                self._phase0_v[pos] = self._quantize_phase(dct_v, 0)
                self._phase1_k[pos] = self._quantize_phase(dct_k, 1)
                self._phase1_v[pos] = self._quantize_phase(dct_v, 1)
        self._pump_count += 1

    def _evict_one(self) -> Optional[int]:
        if not self._entry_positions:
            return None
        oldest = self._entry_positions.pop(0)
        self._phase0_k.pop(oldest, None)
        self._phase0_v.pop(oldest, None)
        self._phase1_k.pop(oldest, None)
        self._phase1_v.pop(oldest, None)
        self._fp32_k.pop(oldest, None)
        self._fp32_v.pop(oldest, None)
        return oldest

    def retrieve(self, position: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            self._retrieval_count += 1
            if position not in self._phase0_k:
                return None
            dct0_k = self._dequantize_phase(*self._phase0_k[position], 0)
            dct0_v = self._dequantize_phase(*self._phase0_v[position], 0)
            dct1_k = self._dequantize_phase(*self._phase1_k[position], 1)
            dct1_v = self._dequantize_phase(*self._phase1_v[position], 1)
            dct_avg_k = (dct0_k + dct1_k) * 0.5
            dct_avg_v = (dct0_v + dct1_v) * 0.5
            k_recovered = idct(dct_avg_k).astype(np.float32)
            v_recovered = idct(dct_avg_v).astype(np.float32)
            if position in self._fp32_k:
                k_orig = self._fp32_k[position]
                v_orig = self._fp32_v[position]
                error_k = float(np.mean((k_recovered - k_orig) ** 2))
                error_v = float(np.mean((v_recovered - v_orig) ** 2))
                self._error_estimate = (
                    0.9 * self._error_estimate + 0.1 * (error_k + error_v) * 0.5
                )
            return k_recovered.reshape(-1), v_recovered.reshape(-1)

    def get_error_cancellation_ratio(self) -> float:
        phase0_only = 0.0
        phase1_only = 0.0
        combined = 0.0
        n = min(10, len(self._entry_positions))
        if n == 0:
            return 1.0
        for pos in list(self._entry_positions)[:n]:
            if pos in self._fp32_k:
                k_orig = self._fp32_k[pos]
                if pos in self._phase0_k:
                    dct0 = self._dequantize_phase(*self._phase0_k[pos], 0)
                    k0 = idct(dct0)
                    phase0_only += float(np.mean((k0 - k_orig.astype(np.float64)) ** 2))
                if pos in self._phase1_k:
                    dct1 = self._dequantize_phase(*self._phase1_k[pos], 1)
                    k1 = idct(dct1)
                    phase1_only += float(np.mean((k1 - k_orig.astype(np.float64)) ** 2))
                if pos in self._phase0_k and pos in self._phase1_k:
                    dct0 = self._dequantize_phase(*self._phase0_k[pos], 0)
                    dct1 = self._dequantize_phase(*self._phase1_k[pos], 1)
                    k_avg = idct((dct0 + dct1) * 0.5)
                    combined += float(np.mean((k_avg - k_orig.astype(np.float64)) ** 2))
        return float(max(phase0_only, phase1_only, EPS) / max(combined, EPS))

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "entry_count": len(self._entry_positions),
                "max_entries": self.max_entries,
                "pump_interval": self.pump_interval,
                "pump_count": self._pump_count,
                "retrieval_count": self._retrieval_count,
                "error_estimate": self._error_estimate,
                "error_cancellation_ratio": self.get_error_cancellation_ratio(),
                "phase_bits": self.phase_bits,
            }


class _SSDStore:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: int) -> str:
        return os.path.join(self.directory, f"kv_{key}.npz")

    def store(self, key: int, k: np.ndarray, v: np.ndarray):
        np.savez_compressed(self._path(key), k=k, v=v)

    def retrieve(self, key: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        path = self._path(key)
        try:
            data = np.load(path)
            return data["k"], data["v"]
        except Exception:
            return None

    def delete(self, key: int):
        path = self._path(key)
        try:
            os.remove(path)
        except OSError:
            pass


@dataclass
class _ExtremeEntry:
    position: int
    head_idx: int
    layer_idx: int
    tier: int
    timestamp: int
    access_count: int
    frequency: float
    last_access_time: int
    importance: float
    k_raw: Optional[np.ndarray] = None
    v_raw: Optional[np.ndarray] = None


class ExtremeTieredKVCache:
    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        head_dim: int = 128,
        n_layers: int = 32,
        max_seq_len: int = 131072,
        ssd_dir: str = "/tmp/spectralstream_extreme_kv",
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.ssd_dir = ssd_dir
        self._tier_capacities = [1024, 8192, 16384, 65536, 262144]
        self._lock = threading.RLock()
        self._next_id = 0
        self._entries: dict[int, _ExtremeEntry] = {}
        self._entry_by_position: dict[tuple[int, int, int], int] = {}
        self._tier_entries: list[dict[int, int]] = [{} for _ in range(5)]
        self._tier0_cache: dict[int, Any] = {}
        self._tier1_compressor = PredictiveCodingKVCache_v2(dim=head_dim)
        self._tier2_cache = TimeCrystalKVCache_v2(
            head_dim=head_dim, max_entries=self._tier_capacities[2]
        )
        self._tier3_cache = SpectralSuperpositionKVCache(
            head_dim=head_dim, max_entries=self._tier_capacities[3]
        )
        self._tier4_ssd = _SSDStore(os.path.join(ssd_dir, "tier4"))
        self._freqkv_compressor = FreqKVExtremeCompressor(head_dim=head_dim)
        self._plasma_evictor = PlasmaConfinementKVEvictor(
            dim=head_dim, n_particles=self._tier_capacities[0]
        )
        self._promotions = 0
        self._demotions = 0
        self._total_stored = 0
        self._total_retrieved = 0
        self._global_step = 0
        self._attention_history: dict[tuple[int, int, int], float] = {}
        self._per_token_bit_budget: dict[int, int] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
        head_idx: int = 0,
        layer_idx: int = 0,
        importance: float = 1.0,
    ) -> int:
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            entry = _ExtremeEntry(
                position=position,
                head_idx=head_idx,
                layer_idx=layer_idx,
                tier=0,
                timestamp=self._global_step,
                access_count=0,
                frequency=1.0,
                last_access_time=self._global_step,
                importance=importance,
            )
            self._per_token_bit_budget[position] = self._dynamic_bit_allocation(
                position, importance
            )
            key_flat = np.ascontiguousarray(key.ravel(), dtype=np.float32)
            value_flat = np.ascontiguousarray(value.ravel(), dtype=np.float32)
            try:
                k_seq = (
                    key_flat.reshape(-1, self.head_dim)
                    if key_flat.size >= self.head_dim
                    else key_flat.reshape(1, -1)
                )
                v_seq = (
                    value_flat.reshape(-1, self.head_dim)
                    if value_flat.size >= self.head_dim
                    else value_flat.reshape(1, -1)
                )
                if k_seq.shape[0] < 4:
                    k_padded = np.zeros((4, k_seq.shape[1]), dtype=np.float32)
                    k_padded[: k_seq.shape[0]] = k_seq
                    v_padded = np.zeros((4, v_seq.shape[1]), dtype=np.float32)
                    v_padded[: v_seq.shape[0]] = v_seq
                    k_seq = k_padded
                    v_seq = v_padded
                compressed = self._freqkv_compressor.compress(
                    np.concatenate([k_seq, v_seq], axis=1)
                )
                entry.k_raw = k_seq[:1].copy()
                entry.v_raw = v_seq[:1].copy()
                self._tier0_cache[eid] = (
                    pickle.dumps(compressed, protocol=pickle.HIGHEST_PROTOCOL),
                    entry,
                )
            except Exception:
                self._tier0_cache[eid] = (key_flat.copy(), value_flat.copy())
            self._entries[eid] = entry
            self._entry_by_position[(layer_idx, head_idx, position)] = eid
            self._tier_entries[0][eid] = eid
            self._total_stored += 1
            self._global_step += 1
            self._plasma_evictor.register_particle(position, importance)
            if len(self._tier_entries[0]) > self._tier_capacities[0]:
                self._demote_to_tier1(eid)
            self._maybe_evict()
            return eid

    def _dynamic_bit_allocation(self, position: int, importance: float) -> int:
        if position < 4 or position > self.max_seq_len - 2:
            return 8
        if importance > 0.9:
            return 6
        if importance > 0.7:
            return 5
        return 4

    def _demote_to_tier1(self, eid: int):
        if eid not in self._entries:
            return
        entry = self._entries[eid]
        self._tier_entries[0].pop(eid, None)
        self._tier_entries[1][eid] = eid
        entry.tier = 1
        self._demotions += 1

    def _demote_to_tier2(self, eid: int):
        if eid not in self._entries:
            return
        entry = self._entries[eid]
        old_tier = entry.tier
        self._tier_entries[old_tier].pop(eid, None)
        self._tier_entries[2][eid] = eid
        entry.tier = 2
        self._demotions += 1
        if entry.k_raw is not None and entry.v_raw is not None:
            self._tier2_cache.store(entry.k_raw, entry.v_raw, entry.position)

    def _demote_to_tier3(self, eid: int):
        if eid not in self._entries:
            return
        entry = self._entries[eid]
        old_tier = entry.tier
        self._tier_entries[old_tier].pop(eid, None)
        self._tier_entries[3][eid] = eid
        entry.tier = 3
        self._demotions += 1
        if entry.k_raw is not None and entry.v_raw is not None:
            self._tier3_cache.store(entry.k_raw, entry.v_raw, entry.position)

    def _demote_to_tier4(self, eid: int):
        if eid not in self._entries:
            return
        entry = self._entries[eid]
        old_tier = entry.tier
        self._tier_entries[old_tier].pop(eid, None)
        self._tier_entries[4][eid] = eid
        entry.tier = 4
        self._demotions += 1
        if entry.k_raw is not None and entry.v_raw is not None:
            self._tier4_ssd.store(eid, entry.k_raw, entry.v_raw)

    def _maybe_evict(self):
        for tier in range(4, -1, -1):
            while len(self._tier_entries[tier]) > self._tier_capacities[tier]:
                entries_list = list(self._tier_entries[tier].keys())
                if not entries_list:
                    break
                oldest_eid = entries_list[0]
                if tier < 4:
                    self._demote_to_tier(tier + 1, oldest_eid)
                else:
                    self._delete_entry(oldest_eid)

    def _demote_to_tier(self, target_tier: int, eid: int):
        {
            0: self._demote_to_tier1,
            1: self._demote_to_tier2,
            2: self._demote_to_tier3,
            3: self._demote_to_tier4,
        }.get(target_tier, lambda x: None)(eid)

    def _delete_entry(self, eid: int):
        if eid not in self._entries:
            return
        entry = self._entries[eid]
        self._tier_entries[entry.tier].pop(eid, None)
        self._entry_by_position.pop(
            (entry.layer_idx, entry.head_idx, entry.position), None
        )
        self._tier0_cache.pop(eid, None)
        self._tier4_ssd.delete(eid)
        self._entries.pop(eid, None)

    def _maybe_promote(self, eid: int):
        if eid not in self._entries:
            return
        entry = self._entries[eid]
        if entry.tier == 0 or entry.frequency <= 5.0:
            return
        old_tier = entry.tier
        self._tier_entries[old_tier].pop(eid, None)
        new_tier = old_tier - 1
        entry.tier = new_tier
        self._tier_entries[new_tier][eid] = eid
        self._promotions += 1
        if new_tier == 0 and entry.k_raw is not None:
            self._tier0_cache[eid] = (entry.k_raw.copy(), entry.v_raw.copy())
        if new_tier == 1:
            self._tier1_compressor._history.append(
                entry.k_raw.ravel().astype(np.float32)
            )
        if new_tier == 2:
            self._tier2_cache.store(entry.k_raw, entry.v_raw, entry.position)
        if new_tier == 3:
            self._tier3_cache.store(entry.k_raw, entry.v_raw, entry.position)

    def retrieve(
        self,
        position: int,
        head_idx: int = 0,
        layer_idx: int = 0,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            key = (layer_idx, head_idx, position)
            eid = self._entry_by_position.get(key)
            if eid is None or eid not in self._entries:
                self._total_retrieved += 1
                return None
            entry = self._entries[eid]
            entry.access_count += 1
            entry.frequency = entry.frequency * 0.9 + 0.1
            entry.last_access_time = self._global_step
            self._plasma_evictor.record_access(position, entry.importance)
            result = self._read_from_tier(entry)
            if result is not None:
                self._maybe_promote(eid)
            self._total_retrieved += 1
            return result

    def _read_from_tier(
        self, entry: _ExtremeEntry
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if entry.tier == 0:
            cached = self._tier0_cache.get(id(entry))
            if cached is not None:
                blob, ent = cached
                if isinstance(blob, bytes):
                    try:
                        compressed = pickle.loads(blob)
                        seq = self._freqkv_compressor.decompress(compressed)
                        return seq[0, : self.head_dim].ravel().astype(np.float32), seq[
                            0, self.head_dim :
                        ].ravel().astype(np.float32)
                    except Exception:
                        pass
            if entry.k_raw is not None:
                return entry.k_raw.ravel().astype(
                    np.float32
                ), entry.v_raw.ravel().astype(np.float32)
            return None
        if entry.tier == 1:
            if entry.k_raw is not None:
                return entry.k_raw.ravel().astype(
                    np.float32
                ), entry.v_raw.ravel().astype(np.float32)
            return None
        if entry.tier == 2:
            result = self._tier2_cache.retrieve(entry.position)
            if result is not None:
                return result[0].ravel().astype(np.float32), result[1].ravel().astype(
                    np.float32
                )
            if entry.k_raw is not None:
                return entry.k_raw.ravel().astype(
                    np.float32
                ), entry.v_raw.ravel().astype(np.float32)
            return None
        if entry.tier == 3:
            result = self._tier3_cache.retrieve(entry.position)
            if result is not None:
                return result[0].ravel().astype(np.float32), result[1].ravel().astype(
                    np.float32
                )
            if entry.k_raw is not None:
                return entry.k_raw.ravel().astype(
                    np.float32
                ), entry.v_raw.ravel().astype(np.float32)
            return None
        if entry.tier == 4:
            result = self._tier4_ssd.retrieve(id(entry))
            if result is not None:
                return result[0].ravel().astype(np.float32), result[1].ravel().astype(
                    np.float32
                )
            result = self._tier4_ssd.retrieve(entry.position)
            if result is not None:
                return result[0].ravel().astype(np.float32), result[1].ravel().astype(
                    np.float32
                )
            return None
        return None

    def query(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        with self._lock:
            q = query_vector.ravel().astype(np.float64)
            q_norm = np.linalg.norm(q) + EPS
            scores = []
            seen_positions = set()
            for tier in range(5):
                for eid in list(self._tier_entries[tier].keys()):
                    if eid not in self._entries:
                        continue
                    entry = self._entries[eid]
                    if entry.position in seen_positions:
                        continue
                    result = self._read_from_tier(entry)
                    if result is None:
                        continue
                    k_vec, _ = result
                    sim = float(np.dot(q, k_vec.ravel().astype(np.float64))) / (
                        q_norm * (np.linalg.norm(k_vec) + EPS)
                    )
                    scores.append((entry.position, sim))
                    seen_positions.add(entry.position)
            scores.sort(key=lambda x: -x[1])
            return scores[:top_k]

    def get_stats(self) -> dict:
        with self._lock:
            per_tier_counts = [len(self._tier_entries[i]) for i in range(5)]
            return {
                "total_entries": sum(per_tier_counts),
                "per_tier_counts": per_tier_counts,
                "tier_capacities": list(self._tier_capacities),
                "promotions": self._promotions,
                "demotions": self._demotions,
                "total_stored": self._total_stored,
                "total_retrieved": self._total_retrieved,
                "global_step": self._global_step,
                "tier0_compression": "FreqKVExtremeCompressor (384:1)",
                "tier1_compression": "PredictiveCodingKVCache_v2 (50:1)",
                "tier2_compression": "TimeCrystalKVCache_v2 (100:1)",
                "tier3_compression": "SpectralSuperpositionKVCache (200:1)",
                "tier4_storage": "SSD (cold)",
                "plasma_evictor": self._plasma_evictor.get_stats(),
                "predictive_coding": self._tier1_compressor.get_stats(),
            }


class KVQualityValidator:
    def __init__(self):
        self._history: list[dict] = []
        self._max_history = 10000
        self._lock = threading.Lock()

    def validate_pair(
        self,
        original_key: np.ndarray,
        retrieved_key: np.ndarray,
        original_value: np.ndarray,
        retrieved_value: np.ndarray,
    ) -> dict:
        o_k = original_key.ravel().astype(np.float64)
        r_k = retrieved_key.ravel().astype(np.float64)
        o_v = original_value.ravel().astype(np.float64)
        r_v = retrieved_value.ravel().astype(np.float64)
        k_cos_sim = float(np.dot(o_k, r_k)) / max(
            np.linalg.norm(o_k) * np.linalg.norm(r_k), EPS
        )
        v_cos_sim = float(np.dot(o_v, r_v)) / max(
            np.linalg.norm(o_v) * np.linalg.norm(r_v), EPS
        )
        k_mse = float(np.mean((o_k - r_k) ** 2))
        v_mse = float(np.mean((o_v - r_v) ** 2))
        k_snr = float(10 * np.log10(np.mean(o_k**2) / max(k_mse, EPS) + EPS))
        v_snr = float(10 * np.log10(np.mean(o_v**2) / max(v_mse, EPS) + EPS))
        k_psnr = float(
            10 * np.log10((np.max(o_k) - np.min(o_k)) ** 2 / max(k_mse, EPS) + EPS)
        )
        v_psnr = float(
            10 * np.log10((np.max(o_v) - np.min(o_v)) ** 2 / max(v_mse, EPS) + EPS)
        )
        k_err_pct = float(np.mean(np.abs(o_k - r_k) / (np.abs(o_k) + EPS)) * 100)
        v_err_pct = float(np.mean(np.abs(o_v - r_v) / (np.abs(o_v) + EPS)) * 100)
        combined_mse = (k_mse + v_mse) * 0.5
        combined_cos = (k_cos_sim + v_cos_sim) * 0.5
        combined_snr = (k_snr + v_snr) * 0.5
        combined_psnr = (k_psnr + v_psnr) * 0.5
        combined_err_pct = (k_err_pct + v_err_pct) * 0.5
        quality = (
            "excellent"
            if combined_cos > 0.98 and combined_err_pct < 0.02
            else "good"
            if combined_cos > 0.95 and combined_err_pct < 0.1
            else "acceptable"
            if combined_cos > 0.9
            else "degraded"
        )
        result = {
            "k_cosine_similarity": k_cos_sim,
            "v_cosine_similarity": v_cos_sim,
            "combined_cosine_similarity": combined_cos,
            "k_mse": k_mse,
            "v_mse": v_mse,
            "combined_mse": combined_mse,
            "k_snr_db": k_snr,
            "v_snr_db": v_snr,
            "combined_snr_db": combined_snr,
            "k_psnr_db": k_psnr,
            "v_psnr_db": v_psnr,
            "combined_psnr_db": combined_psnr,
            "k_error_percent": k_err_pct,
            "v_error_percent": v_err_pct,
            "combined_error_percent": combined_err_pct,
            "retrieval_quality": quality,
        }
        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history.pop(0)
        return result

    def validate_attention_map(
        self, original_attn: np.ndarray, retrieved_attn: np.ndarray
    ) -> dict:
        o = original_attn.ravel().astype(np.float64)
        r = retrieved_attn.ravel().astype(np.float64)
        corr = float(np.corrcoef(o, r)[0, 1]) if len(o) > 1 else 1.0
        mae = float(np.mean(np.abs(o - r)))
        max_err = float(np.max(np.abs(o - r)))
        quality = (
            "excellent"
            if corr > 0.99
            else "good"
            if corr > 0.97
            else "acceptable"
            if corr > 0.95
            else "degraded"
        )
        return {
            "attention_correlation": corr,
            "attention_mae": mae,
            "attention_max_error": max_err,
            "quality": quality,
        }

    def estimate_perplexity_impact(
        self, logprobs_original: np.ndarray, logprobs_retrieved: np.ndarray
    ) -> dict:
        lo = logprobs_original.ravel().astype(np.float64)
        lr = logprobs_retrieved.ravel().astype(np.float64)
        ppl_orig = float(np.exp(-np.mean(lo)))
        ppl_retr = float(np.exp(-np.mean(lr)))
        return {
            "original_perplexity": ppl_orig,
            "retrieved_perplexity": ppl_retr,
            "perplexity_ratio": ppl_retr / max(ppl_orig, EPS),
            "perplexity_delta": ppl_retr - ppl_orig,
            "acceptable": ppl_retr / max(ppl_orig, EPS) < 1.05,
        }

    def batch_validate(
        self, pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> dict:
        results = [self.validate_pair(ok, rk, ov, rv) for ok, rk, ov, rv in pairs]
        return {
            "n_pairs": len(results),
            "avg_cosine_similarity": float(
                np.mean([r["combined_cosine_similarity"] for r in results])
            ),
            "avg_mse": float(np.mean([r["combined_mse"] for r in results])),
            "avg_snr_db": float(np.mean([r["combined_snr_db"] for r in results])),
            "avg_error_percent": float(
                np.mean([r["combined_error_percent"] for r in results])
            ),
            "pass_rate": sum(
                1 for r in results if r["retrieval_quality"] in ("excellent", "good")
            )
            / max(len(results), 1),
        }

    def get_summary(self) -> dict:
        with self._lock:
            if not self._history:
                return {}
            recent = self._history[-100:]
            avg_cos = float(np.mean([r["combined_cosine_similarity"] for r in recent]))
            avg_mse = float(np.mean([r["combined_mse"] for r in recent]))
            avg_snr = float(np.mean([r["combined_snr_db"] for r in recent]))
            avg_err = float(np.mean([r["combined_error_percent"] for r in recent]))
            return {
                "n_validated": len(self._history),
                "recent_avg_cosine_similarity": avg_cos,
                "recent_avg_mse": avg_mse,
                "recent_avg_snr_db": avg_snr,
                "recent_avg_error_percent": avg_err,
                "recent_count": len(recent),
                "pass_rate": sum(
                    1 for r in recent if r["retrieval_quality"] in ("excellent", "good")
                )
                / max(len(recent), 1),
            }

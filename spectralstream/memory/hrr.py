from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

import numpy as np

from spectralstream.core.math_primitives import (
    cascade_eviction_score,
    cosine_similarity,
    generate_random_complex_vector,
    generate_random_hd_vector,
    hrr_bind,
    hrr_bundle,
    hrr_unbind,
    landau_zener_coherence,
    spectral_entropy,
    unit_vector,
)
from spectralstream.kv_cache.core import EPS

logger = logging.getLogger(__name__)


class HRRMode(IntEnum):
    REAL = 0
    COMPLEX = 1


class CacheTier(IntEnum):
    L1 = 0
    L2 = 1
    L3 = 2


def _ensure_dim(x: np.ndarray, dim: int) -> np.ndarray:
    flat = x.ravel().astype(np.float64)
    if len(flat) < dim:
        flat = np.pad(flat, (0, dim - len(flat)))
    elif len(flat) > dim:
        flat = flat[:dim]
    return flat


def _importance_from_entropy(key: np.ndarray, threshold: float = 0.5) -> float:
    ent = spectral_entropy(key)
    if ent < threshold:
        return 0.5 + 0.5 * (ent / max(threshold, 1e-10))
    return 0.5 + 0.5 * min(1.0, ent)


def _eviction_score(
    coherence: float,
    frequency: float,
    importance: float,
) -> float:
    return cascade_eviction_score(
        entropy=1.0 - importance,
        coherence=coherence,
        recency=coherence,
        frequency=frequency,
    )


class HrrMemory:
    def __init__(
        self,
        dim: int = 1024,
        mode: HRRMode = HRRMode.REAL,
        decay_rate: float = 0.0,
        normalise_memory: bool = True,
        resonance_alpha: float = 0.1,
    ):
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"dim must be even and >= 2, got {dim}")
        if dim < 128 or dim > 4096:
            raise ValueError(f"dim must be in [128, 4096], got {dim}")
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in [0, 1], got {decay_rate}")

        self.dim = dim
        self.mode = mode
        self.decay_rate = decay_rate
        self.normalise_memory = normalise_memory
        self.resonance_alpha = resonance_alpha

        if mode == HRRMode.COMPLEX:
            self.memory: np.ndarray = np.zeros(dim, dtype=np.complex128)
        else:
            self.memory = np.zeros(dim, dtype=np.float64)

        self._keys: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        self._timestamps: dict[int, float] = {}
        self._access_count: dict[int, int] = {}
        self._bipolar_rng = np.random.RandomState(0)

    def _normalise(self, v: np.ndarray) -> np.ndarray:
        if self.mode == HRRMode.COMPLEX:
            mag = np.abs(v)
            return v / (mag + EPS)
        return unit_vector(v)

    def generate_key(self, seed: Optional[int] = None) -> np.ndarray:
        if self.mode == HRRMode.COMPLEX:
            return generate_random_complex_vector(self.dim, seed=seed)
        return generate_random_hd_vector(self.dim, seed=seed)

    def _make_flat_key(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(
            seed if seed is not None else self._bipolar_rng.randint(0, 2**31)
        )
        d = self.dim
        d2 = d // 2
        phases = rng.uniform(0, 2 * math.pi, size=d2 + 1)
        phases[0] = rng.choice([0.0, math.pi])
        if d % 2 == 0:
            phases[-1] = rng.choice([0.0, math.pi])
        Y = np.zeros(d, dtype=np.complex128)
        Y[0] = np.cos(phases[0]) + 0j
        for k in range(1, d2):
            Y[k] = np.cos(phases[k]) + 1j * np.sin(phases[k])
        for k in range(d2 + 1, d):
            Y[k] = np.conj(Y[d - k])
        if d % 2 == 0:
            Y[d2] = np.cos(phases[-1]) + 0j
        y = np.fft.ifft(Y).real
        y = y * np.sqrt(d) / np.linalg.norm(y)
        return y.astype(np.float64)

    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.mode == HRRMode.COMPLEX:
            if x.shape != y.shape:
                raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
            return x * y
        return hrr_bind(x, y)

    def unbind(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.mode == HRRMode.COMPLEX:
            if z.shape != y.shape:
                raise ValueError(f"Shape mismatch: {z.shape} vs {y.shape}")
            return np.conj(y) * z
        return hrr_unbind(z, y)

    def bundle(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return hrr_bundle(x, y)

    def bind_direct(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = len(x)
        z = np.zeros(d, dtype=np.float64)
        for k in range(d):
            total = 0.0
            for i in range(d):
                total += float(x[i]) * float(y[(k - i) % d])
            z[k] = total
        return z

    def _frequency_signature(self, v: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.fft(v.astype(np.complex128)))

    def _phase_interference(self, query: np.ndarray, stored: np.ndarray) -> float:
        Q = np.fft.fft(query.astype(np.complex128))
        S = np.fft.fft(stored.astype(np.complex128))
        cross = np.sum(Q * np.conj(S))
        mag_product = np.sqrt(np.sum(np.abs(Q) ** 2) * np.sum(np.abs(S) ** 2))
        if mag_product < 1e-10:
            return 0.0
        return float((cross / mag_product).real)

    def resonance_search(
        self, query: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, float]]:
        if len(self._keys) == 0:
            return []
        if self.mode == HRRMode.COMPLEX:
            q = query.ravel().astype(np.complex128)
            q = q / (np.abs(q) + EPS)
        else:
            q = self._normalise(query.ravel().astype(np.float64))
        scores = []
        for sid, key_vec in self._keys.items():
            score = self._phase_interference(q, key_vec)
            scores.append((sid, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _cosine_search(
        self, query: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, float]]:
        if len(self._keys) == 0:
            return []
        if self.mode == HRRMode.COMPLEX:
            q = query.ravel().astype(np.complex128)
            q = q / (np.abs(q) + EPS)
        else:
            q = self._normalise(query.ravel().astype(np.float64))
        scores = []
        for sid, k_vec in self._keys.items():
            if self.mode == HRRMode.COMPLEX:
                d = len(q)
                sim = float((np.conj(q) * k_vec).real.sum() / d)
            else:
                sim = cosine_similarity(q, k_vec)
            scores.append((sid, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def store(
        self,
        key: np.ndarray,
        value: Optional[np.ndarray] = None,
    ) -> int:
        key = np.asarray(key)
        if key.size == 0:
            raise ValueError("key must have non-zero size")
        if value is not None:
            value = np.asarray(value)
            if value.size == 0:
                raise ValueError("value must have non-zero size if provided")
        try:
            if self.mode == HRRMode.COMPLEX:
                k = key.ravel().astype(np.complex128)
                k = k / (np.abs(k) + EPS)
                if value is not None:
                    v = value.ravel().astype(np.complex128)
                    v = v / (np.abs(v) + EPS)
                    encoded = self.bind(k, v)
                else:
                    encoded = k.copy()
            else:
                k = unit_vector(key.ravel().astype(np.float64))
                if value is not None:
                    v = unit_vector(value.ravel().astype(np.float64))
                    encoded = self.bind(k, v)
                else:
                    encoded = k.copy()

            sid = self._next_id
            self._next_id += 1
            self._keys[sid] = k
            self._timestamps[sid] = time.monotonic()
            self._access_count[sid] = 0

            self.memory = self.bundle(self.memory, encoded)
            if self.normalise_memory:
                self.memory = self._normalise(self.memory)

            return sid
        except (ValueError, FloatingPointError) as e:
            logger.error("HrrMemory.store failed: %s", e)
            raise

    def recall(
        self,
        key: np.ndarray,
        top_k: int = 1,
        use_resonance: bool = True,
    ) -> list[tuple[int, np.ndarray, float]]:
        key = np.asarray(key)
        if key.size == 0:
            raise ValueError("key must have non-zero size")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        try:
            if len(self._keys) == 0:
                return []

            if use_resonance:
                candidates = self.resonance_search(key, top_k=top_k * 2)
            else:
                candidates = self._cosine_search(key, top_k=top_k * 2)

            results = []
            for sid, sim in candidates:
                k_vec = self._keys[sid]
                value_est = self.unbind(self.memory, k_vec)
                self._access_count[sid] = self._access_count.get(sid, 0) + 1
                results.append((sid, value_est, sim))
                if len(results) >= top_k:
                    break
            return results
        except (ValueError, FloatingPointError) as e:
            logger.error("HrrMemory.recall failed: %s", e)
            return []

    def remove(self, storage_id: int) -> bool:
        if storage_id not in self._keys:
            return False
        k_vec = self._keys[storage_id]
        encoded_est = self.bind(k_vec, self.unbind(self.memory, k_vec))
        self.memory = self.memory - self.resonance_alpha * encoded_est
        if self.normalise_memory:
            self.memory = self._normalise(self.memory)
        del self._keys[storage_id]
        self._timestamps.pop(storage_id, None)
        self._access_count.pop(storage_id, None)
        return True

    def clear(self) -> None:
        if self.mode == HRRMode.COMPLEX:
            self.memory = np.zeros(self.dim, dtype=np.complex128)
        else:
            self.memory = np.zeros(self.dim, dtype=np.float64)
        self._keys.clear()
        self._timestamps.clear()
        self._access_count.clear()
        self._next_id = 0

    def apply_decay(self) -> None:
        if self.decay_rate > 0:
            self.memory *= 1.0 - self.decay_rate
            if self.normalise_memory:
                self.memory = self._normalise(self.memory)

    def decay_aged(self, age_threshold: float = 300.0) -> None:
        now = time.monotonic()
        for sid, ts in list(self._timestamps.items()):
            age = now - ts
            if age > age_threshold:
                factor = (1.0 - self.decay_rate) ** (age / age_threshold)
                k_vec = self._keys[sid]
                encoded_est = self.bind(k_vec, self.unbind(self.memory, k_vec))
                self.memory = self.memory - (1.0 - factor) * encoded_est
                if self.normalise_memory:
                    self.memory = self._normalise(self.memory)
                self._timestamps[sid] = now

    @property
    def capacity(self) -> int:
        return max(1, int(self.dim / max(1, math.log(self.dim))))

    def load_factor(self) -> float:
        return len(self._keys) / self.capacity

    def num_items(self) -> int:
        return len(self._keys)

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "mode": self.mode.name,
            "num_items": self.num_items(),
            "capacity": self.capacity,
            "load_factor": self.load_factor(),
            "decay_rate": self.decay_rate,
            "memory_norm": float(np.linalg.norm(self.memory)),
            "normalise": self.normalise_memory,
            "resonance_alpha": self.resonance_alpha,
        }


class FhrrEngine:
    def __init__(self, dim: int = 1024, seed: int = 42):
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.memory: np.ndarray = np.zeros(dim, dtype=np.complex128)
        self._keys: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        self._timestamps: dict[int, float] = {}
        self._access_count: dict[int, int] = {}

    def generate_vector(self, seed: Optional[int] = None) -> np.ndarray:
        return generate_random_complex_vector(self.dim, seed=seed)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        return a * b

    def unbind(self, c: np.ndarray, a: np.ndarray) -> np.ndarray:
        if c.shape != a.shape:
            raise ValueError(f"Shape mismatch: {c.shape} vs {a.shape}")
        return np.conj(a) * c

    def bundle(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        d = len(a)
        return float((np.conj(a) * b).real.sum() / d)

    def store(self, key: np.ndarray, value: np.ndarray) -> int:
        key = np.asarray(key)
        value = np.asarray(value)
        if key.size != self.dim or value.size != self.dim:
            raise ValueError(
                f"Vector dim mismatch: key={len(key)}, value={len(value)}, dim={self.dim}"
            )
        try:
            k = key.ravel().astype(np.complex128)
            v = value.ravel().astype(np.complex128)
            k = k / (np.abs(k) + EPS)
            v = v / (np.abs(v) + EPS)
            encoded = self.bind(k, v)
            self.memory = self.bundle(self.memory, encoded)
            sid = self._next_id
            self._next_id += 1
            self._keys[sid] = k
            self._timestamps[sid] = time.monotonic()
            self._access_count[sid] = 0
            return sid
        except (ValueError, FloatingPointError) as e:
            logger.error("FhrrEngine.store failed: %s", e)
            raise

    def recall(
        self, key: np.ndarray, top_k: int = 1
    ) -> list[tuple[int, np.ndarray, float]]:
        q = key.ravel().astype(np.complex128)
        if not np.allclose(np.abs(q), 1.0, atol=1e-6):
            q = q / np.abs(q)
        scores = []
        for sid, k_vec in self._keys.items():
            sim = self.similarity(q, k_vec)
            scores.append((sid, sim))
        scores.sort(key=lambda x: -x[1])
        results = []
        for sid, sim in scores[:top_k]:
            k_vec = self._keys[sid]
            value_est = self.unbind(self.memory, k_vec)
            self._access_count[sid] = self._access_count.get(sid, 0) + 1
            results.append((sid, value_est, sim))
        return results

    def remove(self, storage_id: int) -> bool:
        if storage_id not in self._keys:
            return False
        k_vec = self._keys[storage_id]
        value_est = self.unbind(self.memory, k_vec)
        encoded_est = self.bind(k_vec, value_est)
        self.memory -= encoded_est / max(len(self._keys), 1)
        del self._keys[storage_id]
        self._timestamps.pop(storage_id, None)
        self._access_count.pop(storage_id, None)
        return True

    def clear(self) -> None:
        self.memory = np.zeros(self.dim, dtype=np.complex128)
        self._keys.clear()
        self._timestamps.clear()
        self._access_count.clear()
        self._next_id = 0

    def apply_decay(self, rate: float = 0.001) -> None:
        if rate > 0:
            self.memory *= 1.0 - rate
            mag = np.abs(self.memory)
            self.memory /= np.max(mag) + EPS

    def get_stats(self) -> dict:
        return {
            "dim": self.dim,
            "num_items": len(self._keys),
            "memory_norm": float(np.linalg.norm(self.memory)),
        }


@dataclass
class ResonantPattern:
    pattern_id: int
    vector: np.ndarray
    frequency_signature: np.ndarray
    phase_profile: np.ndarray
    magnitude_profile: np.ndarray
    timestamp: float
    access_count: int = 0
    coherence: float = 1.0


class ResonantMemory:
    def __init__(
        self,
        dim: int = 1024,
        half_life: float = 1000.0,
        max_patterns: int = 8192,
    ):
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"dim must be even and >= 2, got {dim}")
        self.dim = dim
        self.half_life = half_life
        self.max_patterns = max_patterns
        self._patterns: dict[int, ResonantPattern] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def _decompose(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fv = np.fft.fft(v.astype(np.complex128))
        magnitude = np.abs(fv)
        phase = np.angle(fv)
        return magnitude, phase, fv

    def store(self, vector: np.ndarray) -> int:
        v = unit_vector(vector.ravel().astype(np.float64))
        mag, phase, _ = self._decompose(v)
        sid = self._next_id
        self._next_id += 1
        pattern = ResonantPattern(
            pattern_id=sid,
            vector=v,
            frequency_signature=mag,
            phase_profile=phase,
            magnitude_profile=mag,
            timestamp=time.monotonic(),
        )
        with self._lock:
            self._patterns[sid] = pattern
            self._evict_if_needed()
        return sid

    def retrieve(
        self, query: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, np.ndarray, float]]:
        q = unit_vector(query.ravel().astype(np.float64))
        q_mag, q_phase, q_fv = self._decompose(q)

        now = time.monotonic()
        scores: list[tuple[int, float]] = []

        with self._lock:
            for sid, pat in self._patterns.items():
                age = now - pat.timestamp
                pat.coherence = landau_zener_coherence(age, self.half_life)

                phase_diff = q_phase - pat.phase_profile
                constructive = np.cos(phase_diff)
                weighted = np.sum(q_mag * pat.magnitude_profile * constructive)
                normalisation = (
                    np.sqrt(np.sum(q_mag**2) * np.sum(pat.magnitude_profile**2)) + 1e-10
                )
                score = float(weighted / normalisation) * pat.coherence
                scores.append((sid, score))

        scores.sort(key=lambda x: -x[1])
        results = []
        for sid, score in scores[:top_k]:
            pat = self._patterns[sid]
            pat.access_count += 1
            results.append((sid, pat.vector.copy(), score))
        return results

    def remove(self, pattern_id: int) -> bool:
        with self._lock:
            if pattern_id not in self._patterns:
                return False
            del self._patterns[pattern_id]
            return True

    def apply_decay(self) -> None:
        now = time.monotonic()
        with self._lock:
            for pat in self._patterns.values():
                age = now - pat.timestamp
                pat.coherence = landau_zener_coherence(age, self.half_life)

    def _evict_if_needed(self) -> None:
        while len(self._patterns) > self.max_patterns and len(self._patterns) > 1:
            worst_id = min(
                self._patterns,
                key=lambda k: self._patterns[k].coherence
                * (1.0 + self._patterns[k].access_count),
            )
            del self._patterns[worst_id]

    def num_items(self) -> int:
        return len(self._patterns)

    def clear(self) -> None:
        self._patterns.clear()
        self._next_id = 0

    def get_stats(self) -> dict:
        coherences = [p.coherence for p in self._patterns.values()]
        return {
            "dim": self.dim,
            "num_patterns": len(self._patterns),
            "max_patterns": self.max_patterns,
            "avg_coherence": float(np.mean(coherences)) if coherences else 0.0,
            "half_life": self.half_life,
        }

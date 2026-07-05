from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from spectralstream.core.math_primitives import (
    spectral_entropy,
    landau_zener_coherence,
    cascade_eviction_score,
)
from spectralstream.kv_cache.core import EPS, KVCacheEntry


class EvictionPolicy:
    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        raise NotImplementedError


class SpectralEviction(EvictionPolicy):
    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        positions = np.array([e.position for e in entries], dtype=np.int64)
        global_step = int(positions.max()) + 1 if len(positions) > 0 else 0
        entropies = np.array([spectral_entropy(e.key) for e in entries])
        ages = np.maximum(1, global_step - positions)
        coherences = np.array(
            [landau_zener_coherence(float(a), half_life=1000.0) for a in ages]
        )
        recencies = np.where(global_step > 0, 1.0 - positions / global_step, 0.0)
        freqs = np.array(
            [getattr(e, "score", 1.0) / max(global_step, 1) for e in entries]
        )
        scores = cascade_eviction_score(
            entropy=entropies, coherence=coherences, recency=recencies, frequency=freqs
        )
        return int(np.argmin(scores))


class CascadeEviction(EvictionPolicy):
    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self._attention_scores: Dict[int, float] = {}
        self._entropy_scores: Dict[int, float] = {}
        self._access_times: Dict[int, float] = {}
        self._access_counts: Dict[int, int] = {}

    def record_attention(self, position: int, score: float):
        self._attention_scores[position] = score
        self._access_times[position] = time.monotonic()
        self._access_counts[position] = self._access_counts.get(position, 0) + 1

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

    def eviction_score(self, entry: KVCacheEntry) -> float:
        now = time.monotonic()
        last_access = self._access_times.get(entry.position, now)
        age = now - last_access
        coherence = landau_zener_coherence(float(age), half_life=1000.0)
        entropy = self._entropy_scores.get(entry.position, 0.5)
        recency = 1.0 / (1.0 + age)
        access_count = self._access_counts.get(entry.position, 0)
        frequency = access_count / max(1, access_count + 10)
        return float(
            cascade_eviction_score(
                entropy=np.array([entropy]),
                coherence=np.array([coherence]),
                recency=np.array([recency]),
                frequency=np.array([frequency]),
            )[0]
        )

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        scores = [self.eviction_score(e) for e in entries]
        return int(np.argmin(scores))

    def should_evict(self, entry: KVCacheEntry) -> bool:
        return self.eviction_score(entry) < 0.2

    def evict_positions(
        self, entries: Dict[int, KVCacheEntry], n_to_evict: int
    ) -> List[int]:
        scored = [(pos, self.eviction_score(e)) for pos, e in entries.items()]
        scored.sort(key=lambda x: x[1])
        return [pos for pos, _ in scored[:n_to_evict]]


class H2OEviction(EvictionPolicy):
    def __init__(self, heavy_hitter_frac: float = 0.1):
        self.heavy_hitter_frac = heavy_hitter_frac

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        scores = np.array([e.score for e in entries])
        threshold = float(np.percentile(scores, self.heavy_hitter_frac * 100))
        below = np.where(scores <= threshold)[0]
        if len(below) > 0:
            return int(below[0])
        return int(np.argmin(scores))


class SlidingWindowEviction(EvictionPolicy):
    def __init__(self, window_size: int = 4096):
        self.window_size = window_size

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        positions = np.array([e.position for e in entries])
        return int(np.argmin(positions))


class StreamingLLMEviction(EvictionPolicy):
    def __init__(self, sink_tokens: int = 4, window_size: int = 4096):
        self.sink_tokens = sink_tokens
        self.window_size = window_size

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        candidates = [
            (i, e) for i, e in enumerate(entries) if e.position >= self.sink_tokens
        ]
        if not candidates:
            return -1
        sorted_candidates = sorted(
            candidates, key=lambda x: x[1].position, reverse=True
        )
        keep = self.window_size - self.sink_tokens
        if len(sorted_candidates) <= keep:
            return -1
        return sorted_candidates[0][0]


class ResonanceEviction(EvictionPolicy):
    def __init__(self, window: int = 64):
        self._freq_history: Dict[int, List[float]] = {}

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        scores = np.zeros(len(entries))
        for i, e in enumerate(entries):
            pos = e.position
            if pos not in self._freq_history:
                self._freq_history[pos] = [0.0]
            self._freq_history[pos].append(e.score)
            hist = np.array(self._freq_history[pos][-64:], dtype=np.float64)
            if len(hist) > 1:
                amp = float(np.std(hist))
                freq_dom = float(np.max(np.abs(np.fft.rfft(hist)))) / max(len(hist), 1)
                resonance = amp * freq_dom
            else:
                resonance = 0.0
            scores[i] = resonance
        return int(np.argmin(scores))


class EntropyEviction(EvictionPolicy):
    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        entropies = np.array([spectral_entropy(e.key) for e in entries])
        return int(np.argmin(entropies))


class ImportanceScoring(EvictionPolicy):
    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        positions = np.array([e.position for e in entries], dtype=np.float64)
        scores_arr = np.array([e.score for e in entries], dtype=np.float64)
        max_pos = positions.max() if len(positions) > 0 else 1.0
        frequency = scores_arr / (scores_arr.sum() + EPS)
        recency = positions / (max_pos + EPS)
        attention_sum = np.array(
            [
                float(e.quality.score()) if e.quality is not None else 0.0
                for e in entries
            ]
        )
        importance = 0.4 * frequency + 0.3 * recency + 0.3 * np.abs(attention_sum)
        return int(np.argmin(importance))


class PredictiveEviction(EvictionPolicy):
    def __init__(self, history_window: int = 128):
        self._access_history: np.ndarray = np.zeros(history_window, dtype=np.int64)
        self._history_idx = 0
        self._history_size = history_window

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        positions = np.array([e.position for e in entries], dtype=np.int64)
        n = min(5, len(self._access_history))
        if self._history_idx > n:
            recent = self._access_history[
                max(0, self._history_idx - n) : self._history_idx
            ]
            slope = (
                float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
                if len(recent) > 1
                else 0.0
            )
        else:
            slope = 0.0
        predicted_future = positions + slope
        dist = np.abs(positions - predicted_future)
        dist = dist / (dist.max() + EPS)
        scores_arr = np.array([e.score for e in entries])
        scores_norm = scores_arr / (scores_arr.max() + EPS)
        importance = 0.6 * (1.0 - dist) + 0.4 * scores_norm
        return int(np.argmin(importance))


class ClusteringEviction(EvictionPolicy):
    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        n = len(entries)
        k = min(self.n_clusters, n)
        keys_stack = np.array([e.key.ravel()[:64] for e in entries], dtype=np.float64)
        keys_stack = keys_stack.reshape(n, -1)
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k, n_init=1, random_state=0)
            labels = km.fit_predict(keys_stack)
            cluster_sizes = np.bincount(labels, minlength=k)
            smallest = int(np.argmin(cluster_sizes))
            candidates = np.where(labels == smallest)[0]
            if len(candidates) > 0:
                return int(
                    candidates[np.argmin([entries[c].score for c in candidates])]
                )
        except (ValueError, IndexError, RuntimeError):
            pass
        scores = np.array([e.score for e in entries])
        return int(np.argmin(scores))


class TopologicalEviction(EvictionPolicy):
    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        n = len(entries)
        if n < 3:
            scores = np.array([e.score for e in entries])
            return int(np.argmin(scores))
        keys_stack = np.array([e.key.ravel()[:32] for e in entries], dtype=np.float64)
        keys_stack = keys_stack.reshape(n, -1)
        sim = keys_stack @ keys_stack.T
        norms = np.linalg.norm(sim, axis=-1, keepdims=True)
        sim = sim / (norms + EPS)
        eigvals = np.linalg.eigvalsh(sim)
        persistence = float(np.abs(eigvals[-min(n, 5) :]).sum()) / n
        importance = np.array(
            [e.score * (1.0 + persistence * np.sin(float(e.position))) for e in entries]
        )
        return int(np.argmin(importance))


class ReinforcementLearningEviction(EvictionPolicy):
    def __init__(self, n_arms: int = 4, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values: np.ndarray = np.zeros(n_arms)
        self.arm_counts: np.ndarray = np.zeros(n_arms)
        self._last_action: Optional[int] = None

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        if np.random.random() < self.epsilon:
            arm = int(np.random.randint(self.n_arms))
        else:
            arm = int(np.argmax(self.q_values))
        self.arm_counts[arm] += 1.0
        self._last_action = arm
        n = len(entries)
        if arm == 0:
            return int(np.argmin([e.score for e in entries]))
        elif arm == 1:
            return int(np.argmin([spectral_entropy(e.key) for e in entries]))
        elif arm == 2:
            positions = np.array([e.position for e in entries])
            return int(np.argmax(positions))
        else:
            return int(np.random.randint(n))

    def update(self, hit: bool):
        if self._last_action is None:
            return
        reward = 1.0 if hit else -0.5
        alpha = 1.0 / (self.arm_counts[self._last_action] + 1.0)
        self.q_values[self._last_action] += alpha * (
            reward - self.q_values[self._last_action]
        )
        self._last_action = None


class HybridEviction(EvictionPolicy):
    def __init__(
        self,
        policies: Optional[List[EvictionPolicy]] = None,
        weights: Optional[np.ndarray] = None,
    ):
        if policies is None:
            self.policies: List[EvictionPolicy] = [
                SpectralEviction(),
                H2OEviction(),
                EntropyEviction(),
                ImportanceScoring(),
            ]
        else:
            self.policies = policies
        self.weights = np.ones(len(self.policies)) if weights is None else weights

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        n = len(entries)
        votes = np.zeros(n)
        for i, policy in enumerate(self.policies):
            idx = policy.select_eviction(entries)
            if 0 <= idx < n:
                votes[idx] += self.weights[i]
        if votes.sum() < EPS:
            return int(np.argmin([e.score for e in entries]))
        return int(np.argmax(votes))


class StalenessAwareEviction(EvictionPolicy):
    def __init__(self, half_life: float = 500.0):
        self.half_life = half_life

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        positions = np.array([e.position for e in entries], dtype=np.float64)
        scores_arr = np.array([e.score for e in entries], dtype=np.float64)
        max_pos = positions.max() if len(positions) > 0 else 1.0
        age = max_pos - positions
        decay = 2.0 ** (-age / self.half_life)
        time_value = scores_arr * decay
        return int(np.argmin(time_value))


class AccessPatternEviction(EvictionPolicy):
    def __init__(self, pattern_window: int = 64):
        self._access_buffer: List[int] = []
        self._pattern_window = pattern_window

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        positions = np.array([e.position for e in entries], dtype=np.int64)
        scores_arr = np.array([e.score for e in entries], dtype=np.float64)
        if len(self._access_buffer) > self._pattern_window:
            buf = np.array(self._access_buffer[-self._pattern_window :], dtype=np.int64)
            max_pos = int(positions.max()) + 1 if len(positions) > 0 else 1
            freq = np.zeros(max_pos, dtype=np.float64)
            for p in buf:
                if 0 <= p < max_pos:
                    freq[p] += 1.0
            access_freq = freq[positions]
        else:
            access_freq = scores_arr
        recency = positions / (positions.max() + EPS)
        importance = 0.5 * access_freq + 0.5 * recency
        return int(np.argmin(importance))


class EntropyGradient(EvictionPolicy):
    def __init__(self, window: int = 10):
        self.window = window
        self._entropy_history: Dict[int, List[float]] = {}

    def select_eviction(self, entries: List[KVCacheEntry]) -> int:
        if not entries:
            return -1
        scores = np.zeros(len(entries))
        for i, e in enumerate(entries):
            pos = e.position
            ent = spectral_entropy(e.key)
            if pos not in self._entropy_history:
                self._entropy_history[pos] = []
            self._entropy_history[pos].append(ent)
            hist = self._entropy_history[pos][-self.window :]
            gradient = (
                (hist[-1] - hist[0]) / max(len(hist), 1) if len(hist) > 1 else 0.0
            )
            scores[i] = abs(gradient) * ent
        return int(np.argmin(scores))

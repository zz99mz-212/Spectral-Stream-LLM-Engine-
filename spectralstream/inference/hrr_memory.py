from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


class HrrMemory:
    """Holographic Reduced Representations via circular convolution."""

    def __init__(self, dim: int = 4096, capacity: int = 65536):
        self.dim = dim
        self.capacity = capacity
        self.memory: Dict[int, np.ndarray] = {}

    @staticmethod
    def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        A_fft = np.fft.fft(a.astype(np.complex128))
        B_fft = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(A_fft * B_fft).real.astype(np.float32)

    @staticmethod
    def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        A_fft = np.fft.fft(a.astype(np.complex128))
        B_fft = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(np.conj(A_fft) * B_fft).real.astype(np.float32)

    def _make_key_vector(self, key: int) -> np.ndarray:
        rng = np.random.RandomState(hash(key) & 0x7FFFFFFF)
        vec = rng.randn(self.dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    def store(self, key: int, value: np.ndarray):
        key_vec = self._make_key_vector(key)
        encoded = self._circular_conv(key_vec, value)
        self.memory[key] = encoded
        if len(self.memory) > self.capacity:
            oldest = next(iter(self.memory))
            del self.memory[oldest]

    def recall(self, key: int) -> Optional[np.ndarray]:
        if key not in self.memory:
            return None
        key_vec = self._make_key_vector(key)
        return self._circular_corr(key_vec, self.memory[key])

    def clear(self):
        self.memory.clear()


class HolographicKVCache:
    """KV cache using HRR for compressed associative storage."""

    def __init__(self, dim: int = 128, capacity: int = 4096):
        self.dim = dim
        self.capacity = capacity
        self.hrr = HrrMemory(dim=dim * 2, capacity=capacity)
        self._hit = 0
        self._miss = 0
        self._store_count = 0

    def store(self, position: int, key: np.ndarray, value: np.ndarray):
        k_flat = key.ravel().astype(np.float32)
        v_flat = value.ravel().astype(np.float32)
        combined = np.concatenate([k_flat, v_flat])
        if len(combined) > self.hrr.dim:
            combined = combined[: self.hrr.dim]
        elif len(combined) < self.hrr.dim:
            combined = np.pad(combined, (0, self.hrr.dim - len(combined)))
        self.hrr.store(position, combined)
        self._store_count += 1

    def retrieve(self, position: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        combined = self.hrr.recall(position)
        if combined is None:
            self._miss += 1
            return None
        self._hit += 1
        half = len(combined) // 2
        return combined[:half].copy(), combined[half:].copy()

    def hit_rate(self) -> float:
        total = self._hit + self._miss
        return self._hit / max(total, 1)

    def clear(self):
        self.hrr.clear()
        self._hit = 0
        self._miss = 0
        self._store_count = 0

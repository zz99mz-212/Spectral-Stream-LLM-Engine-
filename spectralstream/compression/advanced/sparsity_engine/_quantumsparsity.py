
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Union

import numpy as np

from spectralstream.core.math_primitives import (
    next_power_of_two as _next_power_of_two,
    softmax as _softmax,
    dct as _dct,
    idct as _idct,
    spectral_entropy as _spectral_entropy,
    fwht,
    ifwht,
)

def _csr_from_dense(dense: np.ndarray, threshold: float = 1e-10) -> tuple:
    mask = np.abs(dense) > threshold
    indices = np.where(mask)
    values = dense[indices]
    return values, indices, dense.shape

def _dense_from_csr(values: np.ndarray, indices: tuple, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    out[indices] = values
    return out

def _nm_mask(shape: tuple, n: int, m: int, rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block_size = block_end - j_block
            chosen = rng.choice(block_size, min(n, block_size), replace=False)
            for c in chosen:
                mask[i, j_block + c] = True
    return mask

def _apply_nm_pattern(weights: np.ndarray, n: int, m: int) -> np.ndarray:
    rows, cols = weights.shape
    out = weights.copy()
    for i in range(rows):
        for j_block in range(0, cols, m):
            block_end = min(j_block + m, cols)
            block = out[i, j_block:block_end]
            if len(block) > n:
                abs_vals = np.abs(block)
                threshold = np.sort(abs_vals)[-n] if n > 0 else 0
                block[abs_vals < threshold] = 0.0
    return out

def _block_mask(shape: tuple, block_h: int, block_w: int, sparsity: float,
                rng_seed: Optional[int] = None) -> np.ndarray:
    rows, cols = shape
    mask = np.ones(shape, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    for i in range(0, rows, block_h):
        for j in range(0, cols, block_w):
            if rng.random() < sparsity:
                ih = min(i + block_h, rows)
                jw = min(j + block_w, cols)
                mask[i:ih, j:jw] = False
    return mask

def _energy_ratio(x: np.ndarray) -> np.ndarray:
    x_spec = _dct(x)
    power = x_spec ** 2
    total = np.sum(power)
    cum = np.cumsum(power) / (total + 1e-30)
    return cum

def _sparsity_ratio(w: np.ndarray) -> float:
    return float(np.mean(np.abs(w) < 1e-10))

def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(A_fft * B_fft).real.astype(np.float64)

def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    A_fft = np.fft.fft(a.astype(np.complex128))
    B_fft = np.fft.fft(b.astype(np.complex128))
    return np.fft.ifft(np.conj(A_fft) * B_fft).real.astype(np.float64)

class QuantumSparsity:
    def __init__(self, n_superpositions: int = 4, dim: int = 512):
        self.n_superpositions = n_superpositions
        self.dim = dim

    def _hadamard(self, n: int) -> np.ndarray:
        H = np.array([[1, 1], [1, -1]], dtype=np.float64) / np.sqrt(2)
        while H.shape[0] < n:
            H = np.kron(H, np.array([[1, 1], [1, -1]], dtype=np.float64) / np.sqrt(2))
        return H[:n, :n]

    def superposition_compress(self, patterns: list[np.ndarray]) -> np.ndarray:
        n = len(patterns)
        d = patterns[0].size
        flat_d = min(d, self.dim)
        amplitudes = np.zeros((n, flat_d), dtype=np.complex128)
        for i, p in enumerate(patterns):
            p_flat = p.ravel()[:flat_d]
            amplitudes[i] = p_flat / (np.linalg.norm(p_flat) + 1e-10)
        H = self._hadamard(n)
        superposition = H @ amplitudes
        return superposition

    def measure_from_superposition(self, superposition: np.ndarray,
                                     n_measurements: int = 1) -> list[np.ndarray]:
        probs = np.abs(superposition) ** 2
        probs = probs / (probs.sum(axis=0, keepdims=True) + 1e-30)
        results = []
        for _ in range(n_measurements):
            measured = np.zeros(superposition.shape[1], dtype=np.float64)
            for i in range(superposition.shape[1]):
                measured[i] = np.random.choice(superposition.shape[0], p=probs[:, i])
            results.append(measured)
        return results

    def represent_sparsity_patterns(self, weights: np.ndarray,
                                     n_patterns: Optional[int] = None) -> np.ndarray:
        if n_patterns is None:
            n_patterns = self.n_superpositions
        patterns = []
        rng = np.random.RandomState(42)
        for _ in range(n_patterns):
            threshold = float(np.percentile(np.abs(weights), rng.uniform(50, 95)))
            pat = (np.abs(weights) >= threshold).astype(np.float64) * weights
            patterns.append(pat)
        return self.superposition_compress(patterns)

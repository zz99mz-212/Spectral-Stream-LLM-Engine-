"""Hyperdimensional vectors and HRR (Holographic Reduced Representations)."""

from typing import Optional

import numpy as np


def generate_random_hd_vector(
    dim: int,
    rng: Optional[np.random.RandomState] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-30)


def generate_random_complex_vector(
    dim: int,
    rng: Optional[np.random.RandomState] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.complex128)
    v.imag = rng.randn(dim)
    return v / (np.linalg.norm(v) + 1e-30)


def hrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = len(a)
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    return np.fft.ifft(A * B).real


def hrr_unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = len(a)
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    return np.fft.ifft(A * np.conj(B)).real


def hrr_bundle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise addition for bundling."""
    return x + y

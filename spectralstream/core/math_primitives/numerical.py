"""Numerical utilities: softmax, logsumexp, unit_vector, cosine_similarity."""

from typing import Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp((x - x_max) / max(temperature, 1e-10))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-30)


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis) + 1e-30)


def gibbs_softmax(
    x: np.ndarray,
    n_samples: int = 1,
    temperature: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState()
    probs = softmax(x, temperature=temperature)
    cum = np.cumsum(probs, axis=-1)
    samples = np.zeros((n_samples,) + x.shape, dtype=np.int64)
    for i in range(n_samples):
        u = rng.uniform(size=x.shape[:-1] + (1,))
        samples[i] = np.argmax(u < cum, axis=-1)
    return samples if n_samples > 1 else samples[0]


def unit_vector(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-30)


def cosine_similarity(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=axis)
    norm_a = np.linalg.norm(a, axis=axis)
    norm_b = np.linalg.norm(b, axis=axis)
    return dot / (norm_a * norm_b + 1e-30)

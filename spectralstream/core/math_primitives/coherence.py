"""Coherence and cascade eviction scoring."""

import numpy as np


def landau_zener_coherence(tensor: np.ndarray, half_life: float = 1000.0) -> float:
    if np.ndim(tensor) == 0:
        age = float(tensor)
        return float(np.exp(-age / max(half_life, 1e-10)))
    flat = tensor.ravel().astype(np.float64)
    if flat.size < 4:
        return 0.0
    flat = flat * np.exp(-np.arange(len(flat), dtype=np.float64) / half_life)
    sorted_vals = np.sort(np.abs(flat))[::-1]
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0
    cumulative = cumulative / total
    n = len(cumulative)
    uniform = np.linspace(1.0 / n, 1.0, n)
    gap = float(np.mean(np.abs(cumulative - uniform)))
    coherence = 1.0 - 2.0 * gap
    return max(0.0, coherence)


def cascade_eviction_score(
    entropy: np.ndarray,
    coherence: np.ndarray,
    recency: np.ndarray,
    frequency: np.ndarray,
) -> np.ndarray:
    entropy = np.asarray(entropy, dtype=np.float64)
    coherence = np.asarray(coherence, dtype=np.float64)
    recency = np.asarray(recency, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64)
    score = (
        0.3 * (1.0 - entropy / (np.max(entropy) + 1e-30))
        + 0.2 * coherence
        + 0.3 * recency
        + 0.2 * (1.0 - frequency / (np.max(frequency) + 1e-30))
    )
    return score

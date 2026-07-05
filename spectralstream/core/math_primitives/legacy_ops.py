"""Legacy numerical ops — migrated from _archive/v1/core/ops.py"""

import numpy as np
from typing import Optional
from spectralstream.core.math_primitives.numerical import softmax


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    return silu_gate * up


def rope(x: np.ndarray, positions: np.ndarray, theta: float = 10000.0) -> np.ndarray:
    n, d = x.shape
    half = d // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    angles = np.outer(positions, freqs)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    x_half1 = x[:, :half]
    x_half2 = x[:, half:]
    rotated = np.concatenate(
        [
            x_half1 * cos - x_half2 * sin,
            x_half1 * sin + x_half2 * cos,
        ],
        axis=-1,
    )
    return rotated


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    d = q.shape[-1]
    scale = scale if scale is not None else 1.0 / np.sqrt(d)
    scores = q @ k.T * scale
    if mask is not None:
        scores = scores + mask[: scores.shape[0], : scores.shape[1]]
    weights = softmax(scores, axis=-1)
    return weights @ v


def attention_tiled(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    tile_size: int = 256,
) -> np.ndarray:
    n_q = q.shape[0]
    d = q.shape[-1]
    output = np.zeros_like(q)
    scale = 1.0 / np.sqrt(d)

    for i in range(0, n_q, tile_size):
        q_tile = q[i : i + tile_size]
        scores = q_tile @ k.T * scale
        weights = softmax(scores, axis=-1)
        output[i : i + tile_size] = weights @ v

    return output


def mean_field_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    n_q = q.shape[0]
    n_k = k.shape[0]
    d = q.shape[-1]

    k_mean = np.mean(k, axis=0, keepdims=True)
    k_var = np.var(k, axis=0, keepdims=True) + 1e-10

    q_dev = q - k_mean
    dist_sq = np.sum(q_dev**2 / (k_var + 1e-10), axis=-1, keepdims=True)

    yukawa = np.exp(-np.sqrt(dist_sq) / sigma) / (1.0 + dist_sq)
    weights = yukawa / (np.sum(yukawa) + 1e-10)

    output = weights * v.mean(axis=0, keepdims=True)

    return output


def top_k_sampling(logits: np.ndarray, k: int = 40, temperature: float = 0.8) -> int:
    probs = softmax(logits, temperature=temperature)
    top_k_indices = np.argpartition(probs, -k)[-k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= np.sum(top_k_probs)
    return int(np.random.choice(top_k_indices, p=top_k_probs))


def min_p_sampling(
    logits: np.ndarray, min_p: float = 0.05, temperature: float = 0.8
) -> int:
    probs = softmax(logits, temperature=temperature)
    max_prob = np.max(probs)
    cutoff = max_prob * min_p
    mask = probs >= cutoff
    if not np.any(mask):
        mask = np.ones_like(probs, dtype=bool)
    filtered_probs = probs * mask
    filtered_probs /= np.sum(filtered_probs)
    return int(np.random.choice(len(filtered_probs), p=filtered_probs))

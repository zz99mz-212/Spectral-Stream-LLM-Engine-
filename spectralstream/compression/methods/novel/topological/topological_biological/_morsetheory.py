from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class MorseTheory:
    """C7. MORSE-THEORY: critical points (maxima/minima/saddles) + integral lines."""

    name = "morse_theory"
    category = "novel_topological"

    def compress(self, tensor: np.ndarray, n_critical: int = 32) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        padded = np.pad(t, 1, mode="edge")
        is_max = (
            (t > padded[:-2, 1:-1])
            & (t > padded[2:, 1:-1])
            & (t > padded[1:-1, :-2])
            & (t > padded[1:-1, 2:])
            & (t > padded[:-2, :-2])
            & (t > padded[:-2, 2:])
            & (t > padded[2:, :-2])
            & (t > padded[2:, 2:])
        )
        is_min = (
            (t < padded[:-2, 1:-1])
            & (t < padded[2:, 1:-1])
            & (t < padded[1:-1, :-2])
            & (t < padded[1:-1, 2:])
            & (t < padded[:-2, :-2])
            & (t < padded[:-2, 2:])
            & (t < padded[2:, :-2])
            & (t < padded[2:, 2:])
        )
        grad_y, grad_x = np.gradient(t)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        is_saddle = (grad_norm > np.percentile(grad_norm, 80)) & ~is_max & ~is_min

        max_idx = np.argwhere(is_max)
        min_idx = np.argwhere(is_min)
        saddle_idx = np.argwhere(is_saddle)

        segment_len = max(1, int(np.sqrt(m * n) * 0.1))

        def _sample_points(idx_arr: np.ndarray, n_max: int) -> np.ndarray:
            if len(idx_arr) == 0:
                return np.zeros((0, 2), dtype=np.int16)
            n_keep = min(len(idx_arr), n_max)
            sel = idx_arr[np.linspace(0, len(idx_arr) - 1, n_keep, dtype=int)]
            return sel.astype(np.int16)

        maxima = _sample_points(max_idx, n_critical // 3)
        minima = _sample_points(min_idx, n_critical // 3)
        saddles = _sample_points(saddle_idx, n_critical // 3)

        meta = dict(
            shape=t.shape, n_max=len(maxima), n_min=len(minima), n_sad=len(saddles)
        )
        data = (
            _serialize(maxima.astype(np.float32))
            + _serialize(minima.astype(np.float32))
            + _serialize(saddles.astype(np.float32))
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_max = metadata["n_max"]
        n_min = metadata["n_min"]
        n_sad = metadata["n_sad"]
        m, n = shape

        pos = 0
        if n_max > 0:
            maxima = _deserialize(data[: n_max * 8]).reshape(-1, 2).astype(int)
            pos = n_max * 8
        else:
            maxima = np.zeros((0, 2), dtype=int)

        if n_min > 0:
            minima = (
                _deserialize(data[pos : pos + n_min * 8]).reshape(-1, 2).astype(int)
            )
            pos += n_min * 8
        else:
            minima = np.zeros((0, 2), dtype=int)

        if n_sad > 0:
            saddles = (
                _deserialize(data[pos : pos + n_sad * 8]).reshape(-1, 2).astype(int)
            )
        else:
            saddles = np.zeros((0, 2), dtype=int)

        recon = np.zeros(shape, dtype=np.float64)
        sigma = min(m, n) * 0.04
        xs = np.arange(m)
        ys = np.arange(n)

        for pts, val in [(maxima, 1.0), (minima, -1.0), (saddles, 0.3)]:
            for pt in pts:
                if pt[0] < m and pt[1] < n:
                    g = np.exp(
                        -((xs - pt[0]) ** 2)[:, None] / (2 * sigma**2)
                        - ((ys - pt[1]) ** 2)[None, :] / (2 * sigma**2)
                    )
                    recon += val * g

        recon = (recon - np.mean(recon)) / (np.std(recon) + 1e-30)
        return recon.astype(np.float32)

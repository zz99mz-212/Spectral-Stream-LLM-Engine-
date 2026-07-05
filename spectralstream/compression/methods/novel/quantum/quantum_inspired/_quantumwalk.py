from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumWalk:
    """Quantum walk on weight graph. Store coin operator + step count.
    Positions correspond to weight values; the walk distribution approximates
    the original distribution.
    """

    name = "quantum_walk"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        n_steps: int = 20,
        n_bins: int = 32,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        t_min, t_max = float(t.min()), float(t.max())
        if abs(t_max - t_min) < 1e-10:
            t_max = t_min + 1.0
        bins = np.linspace(t_min, t_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        hist, _ = np.histogram(t, bins=bins, density=True)
        prob = hist / (hist.sum() + 1e-30)
        coin = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / math.sqrt(2)
        psi = np.zeros((n_bins, 2), dtype=np.complex128)
        psi[:, 0] = np.sqrt(prob) + 0.0j
        for _ in range(n_steps):
            for i in range(n_bins):
                psi[i] = coin @ psi[i]
            new_psi = np.zeros_like(psi)
            for i in range(n_bins):
                if i > 0:
                    new_psi[i - 1, 1] += psi[i, 1]
                if i < n_bins - 1:
                    new_psi[i + 1, 0] += psi[i, 0]
            psi = new_psi
        walk_dist = (np.abs(psi[:, 0]) ** 2 + np.abs(psi[:, 1]) ** 2).real
        walk_dist /= walk_dist.sum() + 1e-30
        bin_f32 = bin_centers.astype(np.float32)
        dist_f32 = walk_dist.astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            n_bins=n_bins,
            t_min=t_min,
            t_max=t_max,
        )
        data = struct.pack("<Iff", n_bins, t_min, t_max)
        data += _serialize(bin_f32)
        data += _serialize(dist_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        n_bins, t_min, t_max = struct.unpack_from("<Iff", data, 0)
        pos = 12
        bin_centers = _deserialize(data[pos : pos + n_bins * 4])
        pos += n_bins * 4
        walk_dist = _deserialize(data[pos : pos + n_bins * 4])
        walk_dist /= walk_dist.sum() + 1e-30
        samples = np.random.choice(n_bins, size=n, p=walk_dist)
        noise = np.random.uniform(
            -0.5 * (t_max - t_min) / n_bins,
            0.5 * (t_max - t_min) / n_bins,
            size=n,
        )
        recon = bin_centers[samples] + noise
        return recon.reshape(shape).astype(np.float32)

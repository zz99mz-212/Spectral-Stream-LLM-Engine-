from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class SpikeNeuralEncoding:
    """D8. SPIKE-NEURAL-ENCODING: r = f(w) = r_max/(1+exp(-β(w-θ))), Poisson."""

    name = "spike_neural_encoding"
    category = "novel_biological"

    def compress(
        self,
        tensor: np.ndarray,
        beta: float = 1.0,
        theta: float = 0.0,
        r_max: float = 100.0,
        n_bins: int = 32,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()

        rate = r_max / (1.0 + np.exp(-beta * (flat - theta)))
        spike_count = np.random.RandomState(0).poisson(rate).astype(np.float64)

        hist, edges = np.histogram(spike_count, bins=n_bins, density=True)
        centers = (edges[:-1] + edges[1:]) * 0.5
        cdf = np.cumsum(hist) / (np.sum(hist) + 1e-30)

        sorted_vals = np.sort(flat)
        quantized = np.interp(cdf, np.linspace(0, 1, len(sorted_vals)), sorted_vals)

        meta = dict(shape=t.shape, n_bins=n_bins, beta=beta, theta=theta, r_max=r_max)
        data = _serialize(quantized.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_bins = metadata["n_bins"]
        N = int(np.prod(shape))

        quantized = _deserialize(data[: n_bins * 4])
        uniform = np.linspace(0, 1, n_bins)
        samples = np.random.RandomState(0).rand(N)
        recon = np.interp(samples, uniform, quantized)
        return recon.reshape(shape).astype(np.float32)

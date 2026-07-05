from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class STDPCodebook:
    """D1. STDP-CODEBOOK: Δw = A_+exp(-Δt/τ_+) for pre→post, A_-exp(-Δt/τ_-) for post→pre."""

    name = "stdp_codebook"
    category = "novel_biological"

    def compress(
        self,
        tensor: np.ndarray,
        n_codebooks: int = 8,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        m, n = t.shape

        flat = t.ravel()
        k = min(n_codebooks, len(flat))

        idx = np.linspace(0, len(flat) - 1, k, dtype=int)
        centroids = flat[idx]

        dt_matrix = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])

        stdp_kernel = np.where(
            dt_matrix >= 0,
            np.exp(-dt_matrix / tau_plus),
            np.exp(-dt_matrix / tau_minus),
        )
        codebook = centroids @ stdp_kernel

        assignments = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)

        meta = dict(shape=t.shape, k=k, tau_plus=tau_plus, tau_minus=tau_minus)
        data = (
            _serialize(codebook.astype(np.float32))
            + assignments.astype(np.int16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        N = int(np.prod(shape))

        codebook = _deserialize(data[: k * 4])
        assignments = np.frombuffer(data[k * 4 :], dtype=np.int16).astype(int)
        if len(assignments) < N:
            assignments = np.resize(assignments, N)
        return codebook[assignments[:N]].reshape(shape).astype(np.float32)

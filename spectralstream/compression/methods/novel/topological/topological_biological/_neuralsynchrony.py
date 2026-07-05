from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class NeuralSynchrony:
    """D12. NEURAL-SYNCHRONY: phase precession φ = 2π·(w-w_min)/(w_max-w_min)."""

    name = "neural_synchrony"
    category = "novel_biological"

    def compress(self, tensor: np.ndarray, n_phases: int = 16) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        w_min, w_max = float(np.min(t)), float(np.max(t))
        w_range = w_max - w_min if w_max > w_min else 1.0

        phase = 2.0 * np.pi * (t - w_min) / w_range
        phase_bins = np.linspace(0, 2 * np.pi, n_phases + 1)
        phase_idx = np.digitize(phase.ravel(), phase_bins) - 1
        phase_idx = np.clip(phase_idx, 0, n_phases - 1)

        amplitude = np.abs(t)
        amp_mean = float(np.mean(amplitude))
        amp_std = float(np.std(amplitude))

        meta = dict(
            shape=t.shape,
            n_phases=n_phases,
            w_min=w_min,
            w_max=w_max,
            amp_mean=amp_mean,
            amp_std=amp_std,
        )
        data = np.array(phase_idx, dtype=np.uint8).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_phases = metadata["n_phases"]
        w_min = metadata["w_min"]
        w_max = metadata["w_max"]
        amp_mean = metadata["amp_mean"]
        amp_std = metadata["amp_std"]
        N = int(np.prod(shape))

        phase_idx = np.frombuffer(data, dtype=np.uint8).astype(int)
        if len(phase_idx) < N:
            phase_idx = np.resize(phase_idx, N)
        phase_idx = np.clip(phase_idx[:N], 0, n_phases - 1)

        phase = phase_idx * (2.0 * np.pi / n_phases)
        recon = w_min + (phase / (2.0 * np.pi)) * (w_max - w_min)
        recon = recon * amp_mean / (np.mean(np.abs(recon)) + 1e-30)
        return recon.reshape(shape).astype(np.float32)

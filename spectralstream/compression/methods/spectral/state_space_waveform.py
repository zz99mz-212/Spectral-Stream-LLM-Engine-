from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StateSpaceWaveform:
    name = "state_space_waveform"
    category = "spectral"

    def __init__(
        self, n_states: int = 8, n_levels: int = 4, n_steps: int = 50, dt: float = 0.1
    ):
        self.n_states = n_states
        self.n_levels = n_levels
        self.n_steps = n_steps
        self.dt = dt

    def _extract_modes(
        self, signal: np.ndarray, n_states: int
    ) -> List[Tuple[float, float, float]]:
        fft_result = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal))
        magnitudes = np.abs(fft_result)

        top_indices = np.argsort(magnitudes)[::-1][:n_states]
        modes = []
        for idx in top_indices:
            freq = float(freqs[idx])
            amp = float(magnitudes[idx]) / len(signal)
            phase = float(np.angle(fft_result[idx]))
            modes.append((amp, freq, phase))
        return modes

    def _synthesize(
        self, modes: List[Tuple[float, float, float]], length: int
    ) -> np.ndarray:
        t = np.arange(length, dtype=np.float64)
        result = np.zeros(length, dtype=np.float64)
        for amp, freq, phase in modes:
            result += amp * np.cos(2 * math.pi * freq * t + phase)
        return result

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_states = kwargs.get("n_states", self.n_states)
        n_levels = kwargs.get("n_levels", self.n_levels)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        m, n = mat.shape

        all_modes = []
        for i in range(m):
            signal = mat[i]
            level_modes = []
            residual = signal.copy()

            for level in range(n_levels):
                chunk_size = max(1, len(residual) // (2**level))
                chunk = residual[:chunk_size]
                modes = self._extract_modes(chunk, n_states)
                level_modes.append(modes)
                reconstructed = self._synthesize(modes, chunk_size)
                residual[:chunk_size] -= reconstructed

            all_modes.append(level_modes)

        data_out = {"modes": all_modes, "n_states": n_states, "n_levels": n_levels}
        meta = {"orig_shape": orig_shape, "method": self.name}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["orig_shape"][-1]
        result = np.zeros((len(data["modes"]), n), dtype=np.float64)

        for i, level_modes in enumerate(data["modes"]):
            signal = np.zeros(n, dtype=np.float64)
            offset = 0
            for level, modes in enumerate(level_modes):
                chunk_size = max(1, n // (2**level))
                chunk_size = min(chunk_size, n - offset)
                if chunk_size <= 0:
                    break
                synthesized = self._synthesize(modes, chunk_size)
                signal[offset : offset + chunk_size] += synthesized
                offset += chunk_size
            result[i] = signal

        return result.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_states = kwargs.get("n_states", self.n_states)
        n_levels = kwargs.get("n_levels", self.n_levels)
        orig = tensor.nbytes
        comp = tensor.shape[0] * n_states * n_levels * 12
        return comp / max(orig, 1)

from __future__ import annotations

import json
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _format_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"

class HolographicWeightEncoder:
    """Encode weight tensors using Holographic Reduced Representations (HRR).

    Uses circular convolution binding to represent weight values as
    superpositions of random phase vectors. Compression is achieved by
    storing only the most significant wave components.
    """

    def __init__(
        self,
        n_waves: int = 64,
        amp_bits: int = 8,
        phase_bits: int = 4,
    ) -> None:
        """
        Args:
            n_waves: Number of wave components to keep.
            amp_bits: Bits per amplitude value.
            phase_bits: Bits per phase value.
        """
        self.n_waves = n_waves
        self.amp_bits = amp_bits
        self.phase_bits = phase_bits

    def compress(self, tensor: np.ndarray) -> dict:
        """Compress tensor using holographic encoding.

        Args:
            tensor: Input tensor (2D preferred).

        Returns:
            Dictionary with wave components and metadata.
        """
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape

        if t.ndim == 1:
            t = t.reshape(1, -1)

        rows, cols = t.shape
        n_waves = min(self.n_waves, rows, cols)

        # FFT decomposition
        fft_result = np.fft.fft2(t)
        # Sort by magnitude and keep top components
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)

        flat_mag = magnitudes.ravel()
        flat_phase = phases.ravel()
        top_indices = np.argsort(flat_mag)[::-1][:n_waves]

        top_amps = flat_mag[top_indices]
        top_phases = flat_phase[top_indices]
        top_rows = top_indices // cols
        top_cols = top_indices % cols

        # Quantize amplitudes and phases
        amp_max = float(np.max(top_amps)) + 1e-10
        amp_levels = (1 << self.amp_bits) - 1
        amp_q = np.round(top_amps / amp_max * amp_levels).astype(np.uint8)

        phase_levels = (1 << self.phase_bits) - 1
        phase_q = np.round((top_phases + np.pi) / (2 * np.pi) * phase_levels).astype(np.uint8)

        return {
            "type": "holographic",
            "orig_shape": list(orig_shape),
            "n_waves": n_waves,
            "amp_bits": self.amp_bits,
            "phase_bits": self.phase_bits,
            "amp_max": amp_max,
            "row_indices": top_rows.astype(np.uint16).tobytes(),
            "col_indices": top_cols.astype(np.uint16).tobytes(),
            "amplitudes": amp_q.tobytes(),
            "phases": phase_q.tobytes(),
            "fft_shape": list(fft_result.shape),
            "n_bytes": (len(top_rows.astype(np.uint16).tobytes()) +
                       len(amp_q.tobytes()) + len(phase_q.tobytes())),
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress holographic weight data.

        Args:
            compressed: Dictionary from compress().

        Returns:
            Reconstructed tensor.
        """
        orig_shape = tuple(compressed["orig_shape"])
        n_waves = compressed["n_waves"]
        amp_bits = compressed.get("amp_bits", 8)
        phase_bits = compressed.get("phase_bits", 4)
        amp_max = compressed.get("amp_max", 1.0)
        fft_shape = tuple(compressed.get("fft_shape", orig_shape))

        row_idx = np.frombuffer(compressed["row_indices"], dtype=np.uint16)
        col_idx = np.frombuffer(compressed["col_indices"], dtype=np.uint16)
        amp_q = np.frombuffer(compressed["amplitudes"], dtype=np.uint8).astype(np.float64)
        phase_q = np.frombuffer(compressed["phases"], dtype=np.uint8).astype(np.float64)

        amp_levels = (1 << amp_bits) - 1
        phase_levels = (1 << phase_bits) - 1

        amplitudes = amp_q / amp_levels * amp_max
        phases = phase_q / phase_levels * 2 * np.pi - np.pi

        # Reconstruct FFT
        fft_recon = np.zeros(fft_shape, dtype=np.complex128)
        for i in range(min(n_waves, len(row_idx), len(col_idx))):
            r, c = int(row_idx[i]), int(col_idx[i])
            if 0 <= r < fft_shape[0] and 0 <= c < fft_shape[1]:
                fft_recon[r, c] = amplitudes[i] * np.exp(1j * phases[i])

        # IFFT
        result = np.fft.ifft2(fft_recon).real
        return result[:orig_shape[0], :orig_shape[1]].astype(np.float32)

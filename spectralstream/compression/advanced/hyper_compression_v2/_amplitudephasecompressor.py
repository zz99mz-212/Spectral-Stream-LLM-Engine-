from __future__ import annotations

import json
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import dct, idct


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


class AmplitudePhaseCompressor:
    """Compress weights using amplitude-phase representation.

    Decomposes each weight element into amplitude |w| and phase sign(w),
    then quantizes amplitude with more bits and phase with fewer bits.
    This exploits the observation that amplitude carries more information
    than sign patterns in neural network weights.
    """

    def __init__(
        self,
        amp_bits: int = 8,
        phase_bits: int = 1,
        keep_energy: float = 0.95,
    ) -> None:
        """
        Args:
            amp_bits: Bit depth for amplitude quantization.
            phase_bits: Bit depth for phase (sign) quantization.
            keep_energy: Energy to retain for spectral pruning.
        """
        self.amp_bits = amp_bits
        self.phase_bits = phase_bits
        self.keep_energy = keep_energy

    def compress(self, tensor: np.ndarray) -> dict:
        """Compress using amplitude-phase encoding.

        Args:
            tensor: Input tensor.

        Returns:
            Dictionary with compressed amplitude, phase, and metadata.
        """
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape
        flat = t.ravel()

        # Separate amplitude and phase
        amplitude = np.abs(flat)
        phase = np.sign(flat)

        # Quantize amplitude
        amp_max = float(np.max(amplitude)) + 1e-10
        n_levels = 1 << self.amp_bits
        amp_normalized = amplitude / amp_max
        amp_quantized = np.round(amp_normalized * (n_levels - 1)).astype(np.uint8)

        # Phase is just sign: 0 for positive, 1 for negative
        phase_packed = np.packbits((phase < 0).astype(np.uint8))

        # Optional: DCT on amplitude for spectral compression
        if len(amplitude) > 64:
            amp_dct = dct(amplitude.astype(np.float64))
            total_e = float(np.sum(amp_dct**2))
            if total_e > 1e-20:
                sorted_mag = np.sort(np.abs(amp_dct))[::-1]
                cumsum = np.cumsum(sorted_mag**2)
                n_keep = int(np.searchsorted(cumsum / total_e, self.keep_energy)) + 1
                n_keep = max(1, min(n_keep, len(amp_dct)))
                amp_dct[n_keep:] = 0
                amplitude_recon = idct(amp_dct).astype(np.float64)
                amp_normalized = amplitude_recon / (
                    np.max(np.abs(amplitude_recon)) + 1e-10
                )
                amp_quantized = np.round(
                    np.clip(amp_normalized, 0, 1) * (n_levels - 1)
                ).astype(np.uint8)

        return {
            "type": "amplitude_phase",
            "orig_shape": list(orig_shape),
            "amp_bits": self.amp_bits,
            "phase_bits": self.phase_bits,
            "amp_max": amp_max,
            "amplitude": amp_quantized.tobytes(),
            "phase": phase_packed.tobytes(),
            "n_elements": len(flat),
            "n_bytes": len(amp_quantized.tobytes()) + len(phase_packed.tobytes()),
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress amplitude-phase data.

        Args:
            compressed: Dictionary from compress().

        Returns:
            Reconstructed tensor.
        """
        orig_shape = tuple(compressed["orig_shape"])
        n_elements = compressed["n_elements"]
        amp_bits = compressed.get("amp_bits", 8)
        amp_max = compressed.get("amp_max", 1.0)
        n_levels = 1 << amp_bits

        amp_raw = np.frombuffer(compressed["amplitude"], dtype=np.uint8).astype(
            np.float64
        )
        amp_normalized = amp_raw / (n_levels - 1)
        amplitude = amp_normalized * amp_max

        phase_bits = compressed.get("phase_bits", 1)
        phase_packed = np.frombuffer(compressed["phase"], dtype=np.uint8)
        phase_bool = np.unpackbits(phase_packed)[:n_elements]
        phase = np.where(phase_bool, -1.0, 1.0)

        result = amplitude[:n_elements] * phase[:n_elements]
        if len(result) < int(np.prod(orig_shape)):
            result = np.pad(result, (0, int(np.prod(orig_shape)) - len(result)))
        return result[: int(np.prod(orig_shape))].reshape(orig_shape).astype(np.float32)

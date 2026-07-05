from __future__ import annotations

import json
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import dct, idct, LloydMaxQuantizer


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


class FrequencySelectiveTD:
    """Compress weights in time domain using frequency-selective filtering.

    Applies a learned filter in frequency domain to keep only the most
    important frequency components of weight vectors.
    """

    def __init__(
        self,
        keep_energy: float = 0.95,
        n_bits: int = 8,
    ) -> None:
        """
        Args:
            keep_energy: Energy fraction to retain.
            n_bits: Quantization bits.
        """
        self.keep_energy = keep_energy
        self.n_bits = n_bits

    def compress(self, tensor: np.ndarray) -> dict:
        """Compress using frequency-selective filtering.

        Args:
            tensor: Input tensor.

        Returns:
            Dictionary with compressed data.
        """
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape
        flat = t.ravel()

        # DCT
        coeffs = dct(flat) if len(flat) > 1 else flat.copy()
        total_energy = float(np.sum(coeffs**2))

        if total_energy < 1e-20:
            return {
                "type": "freq_selective",
                "orig_shape": list(orig_shape),
                "data": b"",
                "n_coeffs": 0,
                "keep_energy": self.keep_energy,
            }

        # Energy-based selection
        sorted_mag = np.sort(np.abs(coeffs))[::-1]
        cumsum = np.cumsum(sorted_mag**2)
        n_keep = int(np.searchsorted(cumsum / total_energy, self.keep_energy)) + 1
        n_keep = max(1, min(n_keep, len(coeffs)))

        top_indices = np.argsort(np.abs(coeffs))[::-1][:n_keep]
        values = coeffs[top_indices]

        # Quantize
        quantizer = LloydMaxQuantizer(self.n_bits)
        quantizer.train(values)
        quantized = quantizer.quantize(values)

        data = json.dumps(
            {
                "indices": top_indices.tolist(),
                "values": quantized.tolist(),
                "total_coeffs": len(coeffs),
            }
        ).encode("utf-8")

        return {
            "type": "freq_selective",
            "orig_shape": list(orig_shape),
            "data": data,
            "n_coeffs": n_keep,
            "keep_energy": self.keep_energy,
            "total_coeffs": len(coeffs),
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress frequency-selective data.

        Args:
            compressed: Dictionary from compress().

        Returns:
            Reconstructed tensor.
        """
        orig_shape = tuple(compressed["orig_shape"])

        if not compressed.get("data"):
            return np.zeros(orig_shape, dtype=np.float32)

        meta = json.loads(compressed["data"])
        total_coeffs = meta.get("total_coeffs", int(np.prod(orig_shape)))

        coeffs = np.zeros(total_coeffs, dtype=np.float64)
        indices = np.array(meta["indices"])
        values = np.array(meta["values"])
        coeffs[indices] = values

        result = idct(coeffs)
        return result[: int(np.prod(orig_shape))].reshape(orig_shape).astype(np.float32)

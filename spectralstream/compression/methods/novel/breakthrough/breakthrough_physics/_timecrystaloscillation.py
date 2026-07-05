from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

class TimeCrystalOscillation:
    """Floquet time crystal: store only the drive parameters (frequency,
    amplitude, phase) that generate the weight via periodic evolution.

    Real implementation: DCT domain keeps top coefficients as 'Floquet modes';
    the time crystal 'drive' is the subset of DCT coefficients that survive
    thresholding. Reconstruction is the inverse DCT (one period of evolution).
    """

    name = "time_crystal_oscillation"
    category = "breakthrough_physics"

    def compress(self, tensor: np.ndarray, n_modes: int = 64) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        # DCT = Floquet drive in frequency domain
        dct_coeffs = np.fft.rfft(flat)
        magnitudes = np.abs(dct_coeffs)
        # Keep top modes (the 'stable time crystal modes')
        k = min(n_modes, len(magnitudes) - 1)
        idx = np.argpartition(magnitudes, -k)[-k:]
        idx = idx[np.argsort(-magnitudes[idx])]
        kept = dct_coeffs[idx]
        # Drive parameters: frequency indices + complex amplitudes
        freq_idx = idx.astype(np.int32)
        buf = struct.pack("<II", n, k)
        buf += freq_idx.tobytes()
        buf += kept.astype(np.complex64).tobytes()
        # Mean (DC component) as 'initial state'
        dc = float(np.mean(flat))
        buf += struct.pack("<d", dc)
        return bytes(buf), {
            "n": n,
            "k": k,
            "dc": dc,
            "shape": tensor.shape,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n, k = struct.unpack_from("<II", data, 0)
        pos = 8
        freq_idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy()
        pos += k * 4
        kept = np.frombuffer(data[pos : pos + k * 8], dtype=np.complex64).copy()
        pos += k * 8
        dc = struct.unpack_from("<d", data, pos)[0]
        spectrum = np.zeros(n // 2 + 1, dtype=np.complex128)
        spectrum[0] = dc * len(kept)
        for idx, val in zip(freq_idx, kept):
            if 0 <= idx < len(spectrum):
                spectrum[idx] = val
        flat = np.fft.irfft(spectrum, n=n)
        return flat.astype(np.float32).reshape(metadata["shape"])

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumPhaseEst:
    """Quantum phase estimation for frequency-domain weight analysis.
    Apply QFT-inspired iterative phase estimation; store phase angles φ_k.
    """

    name = "quantum_phase_est"
    category = "quantum_compression"

    def compress(
        self, tensor: np.ndarray, n_phase_bits: int = 4, top_k: int = 32
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(top_k, n)
        spectrum = np.fft.rfft(t)
        mag = np.abs(spectrum)
        phases = np.angle(spectrum)
        top_idx = np.argsort(-mag)[:k]
        levels = 1 << n_phase_bits
        phase_quant = np.clip(
            np.round((phases[top_idx] / math.pi + 1.0) / 2.0 * levels),
            0,
            levels - 1,
        ).astype(np.uint8)
        mag_norm = mag[top_idx].astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            k=k,
            n_phase_bits=n_phase_bits,
        )
        data = struct.pack("<II", n, k)
        data += top_idx.astype(np.int32).tobytes()
        data += phase_quant.tobytes()
        data += _serialize(mag_norm)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, k = struct.unpack_from("<II", data, 0)
        pos = 8
        top_idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32)
        pos += k * 4
        phase_quant = np.frombuffer(data[pos : pos + k], dtype=np.uint8).astype(
            np.float64
        )
        pos += k
        mag_norm = _deserialize(data[pos : pos + k * 4])
        n_bins = n // 2 + 1
        spectrum = np.zeros(n_bins, dtype=np.complex128)
        levels = 1 << metadata["n_phase_bits"]
        phases = (phase_quant / levels * 2.0 - 1.0) * math.pi
        for i in range(k):
            idx = int(top_idx[i])
            if idx < n_bins:
                spectrum[idx] = mag_norm[i] * complex(
                    math.cos(phases[i]), math.sin(phases[i])
                )
        recon = np.fft.irfft(spectrum, n=n)
        return recon.reshape(shape).astype(np.float32)

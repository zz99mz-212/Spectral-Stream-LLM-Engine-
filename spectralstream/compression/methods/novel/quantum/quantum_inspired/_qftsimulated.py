from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QFTSimulated:
    """Simulated quantum Fourier transform for spectral compression.
    Store only significant QFT coefficients.
    """

    name = "qft_simulated"
    category = "quantum_compression"

    def compress(
        self, tensor: np.ndarray, top_k_ratio: float = 0.15
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        padded = 1 << (n - 1).bit_length()
        data_pad = np.zeros(padded, dtype=np.float64)
        data_pad[:n] = t
        qft = np.fft.fft(data_pad) / math.sqrt(padded)
        mag = np.abs(qft)
        k = max(1, int(padded * top_k_ratio))
        idx = np.argsort(-mag)[:k]
        coeffs = qft[idx].astype(np.complex64)
        meta = dict(
            shape=tensor.shape,
            n=n,
            padded=padded,
            k=k,
        )
        data = struct.pack("<III", n, padded, k)
        data += idx.astype(np.int32).tobytes()
        data += coeffs.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, padded, k = struct.unpack_from("<III", data, 0)
        pos = 12
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32)
        pos += k * 4
        coeffs = np.frombuffer(data[pos : pos + k * 8], dtype=np.complex64)
        qft = np.zeros(padded, dtype=np.complex128)
        for i in range(k):
            qft[idx[i]] = coeffs[i]
        recon = (np.fft.ifft(qft) * math.sqrt(padded)).real[:n]
        return recon.reshape(shape).astype(np.float32)

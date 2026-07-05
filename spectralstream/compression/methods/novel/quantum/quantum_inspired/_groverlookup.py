from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class GroverLookup:
    """Grover's algorithm simulation for O(√N) codebook search.
    Amplitude amplification over codebook candidates. Store oracle
    parameters.
    """

    name = "grover_lookup"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        codebook_size: int = 64,
        n_grover_iters: int = 8,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(codebook_size, n)
        codes = t[np.linspace(0, n - 1, k, dtype=int)].copy()
        assign = np.zeros(n, dtype=np.int32)
        for i in range(n):
            dists = np.abs(t[i] - codes)
            n_states = k
            amplitudes = np.ones(n_states, dtype=np.float64) / math.sqrt(n_states)
            iters = min(n_grover_iters, int(math.pi / 4 * math.sqrt(n_states)))
            for _ in range(iters):
                target_phase = np.exp(
                    1j * math.pi * (dists == dists.min()).astype(float)
                )
                amplitudes = amplitudes * target_phase.real
                mean_amp = amplitudes.mean()
                amplitudes = 2 * mean_amp - amplitudes
                amplitudes = np.clip(amplitudes, 0, None)
                amplitudes /= np.linalg.norm(amplitudes) + 1e-30
            assign[i] = np.argmax(amplitudes)
        codes_f32 = codes.astype(np.float32)
        assign_i16 = assign.astype(np.int16)
        meta = dict(shape=tensor.shape, n=n, k=k)
        data = struct.pack("<II", n, k)
        data += _serialize(codes_f32)
        data += assign_i16.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, k = struct.unpack_from("<II", data, 0)
        pos = 8
        codes = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        assign = np.frombuffer(data[pos : pos + n * 2], dtype=np.int16).astype(np.int32)
        return codes[assign].reshape(shape).astype(np.float32)

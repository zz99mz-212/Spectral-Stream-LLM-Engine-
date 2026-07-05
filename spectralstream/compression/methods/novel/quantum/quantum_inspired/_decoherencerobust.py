from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class DecoherenceRobust:
    """Model decoherence as noise injection. T_1, T_2 relaxation times.
    Apply amplitude damping and dephasing channels as compression.
    """

    name = "decoherence_robust"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        t1_relaxation: float = 0.9,
        t2_dephasing: float = 0.8,
        block_size: int = 32,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        damped = t * math.sqrt(t1_relaxation)
        dephased = damped * math.sqrt(t2_dephasing) + np.random.randn(n) * math.sqrt(
            1.0 - t2_dephasing
        )
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float64)
        padded[:n] = dephased
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
        quantized = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)
        meta = dict(
            shape=tensor.shape,
            n=n,
            t1=t1_relaxation,
            t2=t2_dephasing,
            block_size=block_size,
        )
        data = struct.pack("<IffI", n, t1_relaxation, t2_dephasing, block_size)
        data += scales.astype(np.float32).tobytes()
        data += quantized.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, t1, t2, block_size = struct.unpack_from("<IffI", data, 0)
        pos = 16
        n_blocks = int(math.ceil(n / block_size))
        scales = np.frombuffer(
            data[pos : pos + n_blocks * 4], dtype=np.float32
        ).reshape(-1, 1)
        pos += n_blocks * 4
        quantized = (
            np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
            .reshape(-1, block_size)
            .astype(np.float32)
        )
        dephased = (quantized * scales).ravel()[:n]
        t1_correct = t1 if t1 > 0 else 0.5
        recon = dephased / (math.sqrt(t1_correct * t2) + 1e-10)
        return recon.reshape(shape).astype(np.float32)

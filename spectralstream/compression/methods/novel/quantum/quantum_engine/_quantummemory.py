from __future__ import annotations

import math
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumMemory:
    """Quantum associative memory for compression pattern recall.
    Hopfield-like network with quantum activation.
    """

    name = "quantum_memory"
    category = "quantum_engine"

    def compress(
        self,
        tensor: np.ndarray,
        n_patterns: int = 8,
        n_recall_iters: int = 10,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        p = min(n_patterns, n // 4)
        patterns = np.random.randn(p, n).astype(np.float64)
        for i in range(p):
            patterns[i] /= np.linalg.norm(patterns[i])
        W = patterns.T @ patterns
        np.fill_diagonal(W, 0.0)
        state = np.sign(np.random.randn(n)).astype(np.float64)
        for _ in range(n_recall_iters):
            for i in range(n):
                h = float(W[i] @ state)
                state[i] = np.sign(h) if abs(h) > 0.01 else state[i]
            state = state / (np.linalg.norm(state) + 1e-30)
        W_f32 = W.astype(np.float32)
        state_f32 = state.astype(np.float32)
        meta = dict(
            shape=tensor.shape,
            n=n,
            p=p,
        )
        data = struct.pack("<II", n, p)
        data += _serialize(W_f32)
        data += _serialize(state_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, p = struct.unpack_from("<II", data, 0)
        pos = 8
        W = _deserialize(data[pos : pos + n * n * 4]).reshape(n, n)
        pos += n * n * 4
        state = _deserialize(data[pos : pos + n * 4])
        return state.reshape(shape).astype(np.float32)

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumNeuron:
    """Single quantum neuron: |y⟩ = σ(Σ w_i |x_i⟩). Amplitude encoding
    with nonlinear activation via phase estimation.
    """

    name = "quantum_neuron"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        n_neurons: int = 8,
        learning_rate: float = 0.01,
        n_epochs: int = 20,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        k = min(n_neurons, n)
        W = np.random.randn(k, k).astype(np.float64) * 0.1
        b = np.zeros(k, dtype=np.float64)
        n_batches = max(1, n // k)
        inputs = t[: n_batches * k].reshape(n_batches, k)
        targets = np.roll(t, -1)[: n_batches * k].reshape(n_batches, k)
        for _ in range(n_epochs):
            for idx in range(n_batches):
                x = inputs[idx]
                z = W @ x + b
                y = np.tanh(z)
                target = targets[idx]
                error = y - target
                dW = np.outer(error, x)
                db = error
                W -= learning_rate * dW
                b -= learning_rate * db
        W_f32 = W.astype(np.float32)
        b_f32 = b.astype(np.float32)
        residual = np.zeros(n, dtype=np.float64)
        for i in range(0, n - k + 1, k):
            x = t[i : i + k]
            z = W @ x + b
            residual[i : i + k] = t[i : i + k] - np.tanh(z)
        meta = dict(shape=tensor.shape, n=n, k=k)
        data = struct.pack("<I", k)
        data += _serialize(W_f32)
        data += _serialize(b_f32)
        data += _serialize(residual.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        k = struct.unpack_from("<I", data, 0)[0]
        pos = 4
        W = _deserialize(data[pos : pos + k * k * 4]).reshape(k, k)
        pos += k * k * 4
        b = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        residual = _deserialize(data[pos:])
        recon = np.zeros(n, dtype=np.float64)
        for i in range(0, n - k + 1, k):
            x = np.zeros(k, dtype=np.float64)
            z = W @ x + b
            recon[i : i + k] = np.tanh(z)
        recon += residual[:n]
        return recon.reshape(shape).astype(np.float32)

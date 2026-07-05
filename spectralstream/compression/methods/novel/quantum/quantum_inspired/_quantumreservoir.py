from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumReservoir:
    """Quantum reservoir computing: random quantum circuit + linear readout.
    Use random unitary matrices as reservoir; train linear readout via ridge regression.
    """

    name = "quantum_reservoir"
    category = "quantum_compression"

    def __init__(self) -> None:
        self._rng = np.random.RandomState(42)

    def _random_unitary(self, d: int) -> np.ndarray:
        A = self._rng.randn(d, d).astype(np.float64) + 1j * self._rng.randn(
            d, d
        ).astype(np.float64)
        Q, _ = np.linalg.qr(A)
        return Q

    def compress(
        self,
        tensor: np.ndarray,
        reservoir_dim: int = 32,
        n_layers: int = 3,
        ridge_alpha: float = 0.01,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64).ravel()
        n = len(t)
        d = min(reservoir_dim, n)
        units = [self._random_unitary(d) for _ in range(n_layers)]
        state = np.zeros(d, dtype=np.complex128)
        states = np.zeros((n, d), dtype=np.complex128)
        for i in range(n):
            state += t[i]
            for layer in range(n_layers):
                state = units[layer] @ state
            state = state / (np.linalg.norm(state) + 1e-30)
            states[i] = state
        X = np.column_stack([states.real, states.imag])
        w = np.linalg.solve(X.T @ X + ridge_alpha * np.eye(2 * d), X.T @ t)
        y_pred = X @ w
        residual = t - y_pred
        meta = dict(
            shape=tensor.shape,
            n=n,
            d=d,
            n_layers=n_layers,
            ridge_alpha=ridge_alpha,
        )
        data = struct.pack("<II", d, n_layers)
        for u in units:
            data += u.astype(np.complex64).tobytes()
        data += _serialize(w.astype(np.float32))
        data += _serialize(residual.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n"]
        d, n_layers = struct.unpack_from("<II", data, 0)
        pos = 8
        units = []
        for _ in range(n_layers):
            u = np.frombuffer(data[pos : pos + d * d * 8], dtype=np.complex64).reshape(
                d, d
            )
            pos += d * d * 8
            units.append(u)
        w = _deserialize(data[pos : pos + 2 * d * 4])
        pos += 2 * d * 4
        residual = _deserialize(data[pos:])
        state = np.zeros(d, dtype=np.complex128)
        recon = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for layer in range(n_layers):
                state = units[layer].astype(np.complex128) @ state
            state = state / (np.linalg.norm(state) + 1e-30)
            x_i = np.concatenate([state.real, state.imag])
            recon[i] = x_i @ w
            state += recon[i]
        recon += residual
        return recon.reshape(shape).astype(np.float32)

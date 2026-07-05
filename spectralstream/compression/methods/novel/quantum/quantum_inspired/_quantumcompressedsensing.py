from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumCompressedSensing:
    """Quantum compressed sensing: matrix completion via Pauli measurements.
    Use random Pauli-like projections to recover low-rank structure.
    """

    name = "quantum_compressed_sensing"
    category = "quantum_compression"

    def compress(
        self, tensor: np.ndarray, n_measurements: int = 64, rank: int = 4
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        flat = t.ravel()
        n = len(flat)
        m = min(n_measurements, n)
        r = min(rank, m // 2)
        rng = np.random.RandomState(42)
        pauli_basis = rng.choice([-1.0, 1.0], size=(m, n)).astype(np.float64)
        measurements = pauli_basis @ flat
        U, S, Vt = np.linalg.svd(measurements.reshape(m // 2, 2), full_matrices=False)
        k = min(r, len(S))
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        pauli_basis_f32 = pauli_basis.astype(np.float32)
        meas_f32 = measurements.astype(np.float32)
        U_k_f32 = U_k.astype(np.float32)
        S_k_f32 = S_k.astype(np.float32)
        Vt_k_f32 = Vt_k.astype(np.float32)
        meta = dict(
            shape=orig_shape,
            n=n,
            m=m,
            k=k,
        )
        data = struct.pack("<III", n, m, k)
        data += pauli_basis_f32.tobytes()
        data += _serialize(meas_f32)
        data += _serialize(U_k_f32)
        data += _serialize(S_k_f32)
        data += _serialize(Vt_k_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, m, k = struct.unpack_from("<III", data, 0)
        pos = 12
        pauli_basis = np.frombuffer(
            data[pos : pos + m * n * 4], dtype=np.float32
        ).reshape(m, n)
        pos += m * n * 4
        measurements = _deserialize(data[pos : pos + m * 4])
        pos += m * 4
        U_k = _deserialize(data[pos : pos + (m // 2) * k * 4]).reshape(m // 2, k)
        pos += (m // 2) * k * 4
        S_k = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        Vt_k = _deserialize(data[pos : pos + k * 2 * 4]).reshape(k, 2)
        pauli_inv = (
            np.linalg.pinv(pauli_basis.T @ pauli_basis + 1e-10 * np.eye(n))
            @ pauli_basis.T
        )
        est = pauli_inv @ measurements
        return est.reshape(shape).astype(np.float32)

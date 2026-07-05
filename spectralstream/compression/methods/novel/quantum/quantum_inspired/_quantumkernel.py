from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumKernel:
    """Quantum kernel: K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|².
    Fidelity estimation via swap test simulation. Use kernel PCA for compression.
    """

    name = "quantum_kernel"
    category = "quantum_compression"

    def _quantum_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-30)
        b_n = b / (np.linalg.norm(b) + 1e-30)
        overlap = abs(np.dot(a_n, b_n.conj())) ** 2
        return float(overlap)

    def compress(self, tensor: np.ndarray, n_components: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        flat = t.ravel()
        n = len(flat)
        k = min(n_components, n // 2)
        idx = np.linspace(0, n - 1, k, dtype=int)
        basis_vecs = flat[idx]
        K = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            v1 = flat[:k]
            for j in range(k):
                K[i, j] = self._quantum_fidelity(v1, basis_vecs)
        eigvals, eigvecs = np.linalg.eigh(K)
        order = np.argsort(-eigvals)
        eigvals = eigvals[order][:k]
        eigvecs = eigvecs[:, order][:, :k]
        proj = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            v = flat[:k]
            for j in range(k):
                bv = np.full(k, basis_vecs[j])
                proj[i, j] = self._quantum_fidelity(v, bv)
        coeffs = proj @ eigvecs
        coeffs_f32 = coeffs.astype(np.float32)
        eigvecs_f32 = eigvecs.astype(np.float32)
        eigvals_f32 = eigvals.astype(np.float32)
        meta = dict(shape=orig_shape, n=n, k=k)
        data = struct.pack("<II", n, k)
        data += _serialize(coeffs_f32)
        data += _serialize(eigvecs_f32)
        data += _serialize(eigvals_f32)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n, k = struct.unpack_from("<II", data, 0)
        pos = 8
        coeffs = _deserialize(data[pos : pos + n * k * 4]).reshape(n, k)
        pos += n * k * 4
        eigvecs = _deserialize(data[pos : pos + k * k * 4]).reshape(k, k)
        pos += k * k * 4
        eigvals = _deserialize(data[pos : pos + k * 4])
        recon = (coeffs @ eigvecs.T)[:, 0]
        scale = np.sqrt(eigvals.mean()) / (recon.std() + 1e-30)
        return (recon * scale).reshape(shape).astype(np.float32)

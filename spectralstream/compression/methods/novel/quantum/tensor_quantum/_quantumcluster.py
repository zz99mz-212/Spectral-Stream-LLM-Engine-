from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumCluster:
    name = "quantum_cluster"
    category = "tensor_quantum"

    _PAULIS = (
        np.eye(2, dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    )

    def compress(self, tensor: np.ndarray, cluster_size: int = 2) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            nq = max(1, int(math.ceil(math.log2(max(m, n)))))
            dim = 1 << nq
            t_pad = np.zeros((dim, dim), dtype=np.complex128)
            t_pad[:m, :n] = t
            n_terms = min(4**nq, 64)
            coeffs = np.zeros(n_terms, dtype=np.complex128)
            for s in range(n_terms):
                op = np.array([[1]], dtype=np.complex128)
                for q in range(nq):
                    p_idx = (s >> (2 * q)) & 3
                    op = np.kron(op, self._PAULIS[p_idx])
                coeffs[s] = np.trace(op.conj().T @ t_pad) / dim
            thr = np.percentile(
                np.abs(coeffs), 100 - max(5, 100 // max(1, cluster_size))
            )
            mask = np.abs(coeffs) >= thr
            sidx = np.where(mask)[0].astype(np.int16)
            svals = coeffs[mask].astype(np.complex64)
            meta = dict(
                shape=orig_shape,
                cluster_size=cluster_size,
                n_qubits=nq,
                n_terms=len(sidx),
            )
            data = sidx.tobytes() + svals.tobytes()
            return data, meta
        meta = dict(shape=orig_shape, cluster_size=0, n_qubits=0, n_terms=0)
        data = _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_terms = metadata["n_terms"]
        if n_terms == 0:
            flat = _deserialize(data)
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        nq = metadata["n_qubits"]
        sidx = np.frombuffer(data[: n_terms * 2], dtype=np.int16)
        svals = np.frombuffer(data[n_terms * 2 :], dtype=np.complex64).astype(
            np.complex128
        )
        dim = 1 << nq
        recon = np.zeros((dim, dim), dtype=np.complex128)
        for s, val in zip(sidx, svals):
            op = np.array([[1]], dtype=np.complex128)
            for q in range(nq):
                p_idx = (s >> (2 * q)) & 3
                op = np.kron(op, self._PAULIS[p_idx])
            recon += val * op
        m, n = shape
        return np.real(recon[:m, :n]).astype(np.float32)

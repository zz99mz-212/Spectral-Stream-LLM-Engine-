from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumCircuit:
    name = "quantum_circuit"
    category = "tensor_quantum"

    def _rx(self, theta: float) -> np.ndarray:
        c, s = math.cos(theta * 0.5), math.sin(theta * 0.5)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    def _ry(self, theta: float) -> np.ndarray:
        c, s = math.cos(theta * 0.5), math.sin(theta * 0.5)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    def _rz(self, theta: float) -> np.ndarray:
        return np.array(
            [
                [math.cos(theta * 0.5) - 1j * math.sin(theta * 0.5), 0],
                [0, math.cos(theta * 0.5) + 1j * math.sin(theta * 0.5)],
            ],
            dtype=np.complex128,
        )

    def compress(self, tensor: np.ndarray, n_layers: int = 3) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            n_elem = m * n
            nq = max(1, int(math.ceil(math.log2(n_elem))))
            dim = 1 << nq
            thetas = np.random.randn(n_layers, nq, 3).astype(np.float32)
            U_full = np.eye(dim, dtype=np.complex128)
            for layer in range(n_layers):
                U_layer = np.eye(dim, dtype=np.complex128)
                for q in range(nq):
                    single = (
                        self._rz(thetas[layer, q, 2])
                        @ self._ry(thetas[layer, q, 1])
                        @ self._rx(thetas[layer, q, 0])
                    )
                    op = np.eye(dim, dtype=np.complex128)
                    for i in range(dim):
                        b = (i >> q) & 1
                        for j in range(dim):
                            bj = (j >> q) & 1
                            if (i & ~(1 << q)) == (j & ~(1 << q)):
                                op[i, j] = single[b, bj]
                    U_layer = U_layer @ op
                for q in range(nq - 1):
                    cnot = np.eye(dim, dtype=np.complex128)
                    for i in range(dim):
                        control = (i >> q) & 1
                        if control:
                            j = i ^ (1 << (q + 1))
                            cnot[i, j] = 1
                            cnot[i, i] = 0
                    U_layer = U_layer @ cnot
                U_full = U_full @ U_layer
            rho = U_full[:, :1] @ U_full[:, :1].conj().T
            diag = np.real(np.diag(rho))
            scale = np.dot(diag[:n_elem], t.ravel()) / (
                np.dot(diag[:n_elem], diag[:n_elem]) + 1e-30
            )
            meta = dict(
                shape=orig_shape,
                n_layers=n_layers,
                n_qubits=nq,
                dim=dim,
                theta_shape=thetas.shape,
                scale=float(scale),
            )
            data = _serialize(thetas)
            return data, meta
        meta = dict(
            shape=orig_shape, n_layers=0, n_qubits=0, theta_shape=(0,), scale=0.0
        )
        data = _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_layers = metadata["n_layers"]
        if n_layers == 0:
            flat = _deserialize(data)
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        nq = metadata["n_qubits"]
        tshape = metadata["theta_shape"]
        sz = int(np.prod(tshape))
        thetas = _deserialize(data[: sz * 4]).reshape(tshape)
        dim = 1 << nq
        U_full = np.eye(dim, dtype=np.complex128)
        for layer in range(n_layers):
            U_layer = np.eye(dim, dtype=np.complex128)
            for q in range(nq):
                single = (
                    self._rz(thetas[layer, q, 2])
                    @ self._ry(thetas[layer, q, 1])
                    @ self._rx(thetas[layer, q, 0])
                )
                op = np.eye(dim, dtype=np.complex128)
                for i in range(dim):
                    b = (i >> q) & 1
                    for j in range(dim):
                        bj = (j >> q) & 1
                        if (i & ~(1 << q)) == (j & ~(1 << q)):
                            op[i, j] = single[b, bj]
                U_layer = U_layer @ op
            for q in range(nq - 1):
                cnot = np.eye(dim, dtype=np.complex128)
                for i in range(dim):
                    control = (i >> q) & 1
                    if control:
                        j = i ^ (1 << (q + 1))
                        cnot[i, j] = 1
                        cnot[i, i] = 0
                U_layer = U_layer @ cnot
            U_full = U_full @ U_layer
        rho = U_full[:, :1] @ U_full[:, :1].conj().T
        diag = np.real(np.diag(rho))
        n_elem = int(np.prod(shape))
        if len(diag) < n_elem:
            diag = np.pad(diag, (0, n_elem - len(diag)))
        return (diag[:n_elem] * metadata["scale"]).reshape(shape).astype(np.float32)

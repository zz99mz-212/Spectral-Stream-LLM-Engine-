from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class KoopmanOperator:
    """Koopman operator: Kf(x) = f(F(x)), DMD via SVD eigenfunctions."""

    name = "koopman_operator"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        X = t[:, :-1]
        Y = t[:, 1:]
        k = min(rank, min(m, n - 1))

        Ux, Sx, Vtx = np.linalg.svd(X, full_matrices=False)
        r = min(k, len(Sx))
        U_r = Ux[:, :r]
        S_r_inv = np.diag(1.0 / (Sx[:r] + 1e-30))
        V_r = Vtx[:r, :]

        K_tilde = U_r.T @ Y @ V_r.T @ S_r_inv

        eigvals, eigvecs = np.linalg.eig(K_tilde)
        phi = Y @ V_r.T @ S_r_inv @ eigvecs

        meta = dict(shape=tensor.shape, rank=r)
        data = _serialize(eigvals.astype(np.complex64).real.astype(np.float32))
        data += _serialize(eigvals.astype(np.complex64).imag.astype(np.float32))
        data += _serialize(phi.real.astype(np.float32))
        data += _serialize(U_r.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]

        pos = 0
        eig_real = _deserialize(data[: rank * 4])
        pos += rank * 4
        eig_imag = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        eigvals = eig_real + 1j * eig_imag

        phi = _deserialize(data[pos : pos + shape[-1] * rank * 4]).reshape(
            shape[-1], rank
        )
        pos += shape[-1] * rank * 4
        U_r = _deserialize(data[pos : pos + shape[0] * rank * 4]).reshape(
            shape[0], rank
        )

        rng = np.random.RandomState(42)
        b = rng.randn(rank)
        time = np.arange(shape[-1])
        V = np.exp(np.outer(time, eigvals))
        recon = (phi.real * b[np.newaxis, :]) @ V.real.T
        return recon.reshape(shape).astype(np.float32)

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class MatrixProductOperator:
    name = "matrix_product_operator"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, bond_dim: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            chi = min(bond_dim, m, n)
            u, s, vt = np.linalg.svd(t, full_matrices=False)
            rk = min(chi, len(s))
            A = (u[:, :rk] * s[:rk]).copy()
            B = vt[:rk, :].copy()
            meta = dict(
                shape=orig_shape,
                bond_dim=chi,
                rank=rk,
                A_shape=A.shape,
                B_shape=B.shape,
            )
            data = (
                struct.pack("<iii", chi, rk, 2)
                + _serialize(A.astype(np.float32))
                + _serialize(B.astype(np.float32))
            )
            return data, meta
        meta = dict(
            shape=orig_shape, bond_dim=bond_dim, rank=0, A_shape=(0, 0), B_shape=(0, 0)
        )
        data = struct.pack("<iii", bond_dim, 0, 0) + _serialize(
            t.ravel().astype(np.float32)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bond_dim, rk, n_cores = struct.unpack_from("<iii", data, 0)
        if rk == 0:
            flat = _deserialize(data[12:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 12
        A = _deserialize(
            data[pos : pos + int(np.prod(metadata["A_shape"])) * 4]
        ).reshape(metadata["A_shape"])
        pos += int(np.prod(metadata["A_shape"])) * 4
        B = _deserialize(
            data[pos : pos + int(np.prod(metadata["B_shape"])) * 4]
        ).reshape(metadata["B_shape"])
        return (A @ B).reshape(shape).astype(np.float32)

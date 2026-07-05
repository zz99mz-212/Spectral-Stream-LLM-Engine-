from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class TTCross:
    name = "tt_cross"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            r = min(rank, m, n)
            row_norms = np.sum(t**2, axis=1) + 1e-30
            col_norms = np.sum(t**2, axis=0) + 1e-30
            ridx = np.sort(
                np.random.choice(
                    m, size=r, replace=False, p=row_norms / row_norms.sum()
                )
            )
            cidx = np.sort(
                np.random.choice(
                    n, size=r, replace=False, p=col_norms / col_norms.sum()
                )
            )
            C = t[ridx, :]
            R = t[:, cidx]
            U_mat = t[np.ix_(ridx, cidx)]
            U_inv = np.linalg.pinv(U_mat + 1e-10 * np.eye(r))
            cores = [
                C.astype(np.float32),
                U_inv.astype(np.float32),
                R.astype(np.float32),
            ]
            meta = dict(
                shape=orig_shape,
                rank=r,
                n_cores=3,
                core_shapes=[c.shape for c in cores],
            )
            data = struct.pack("<ii", r, 3) + b"".join(_serialize(c) for c in cores)
            return data, meta
        meta = dict(shape=orig_shape, rank=rank, n_cores=0, core_shapes=[])
        data = struct.pack("<ii", rank, 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r, n_cores = struct.unpack_from("<ii", data, 0)
        if n_cores == 0:
            flat = _deserialize(data[8:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 8
        cores = []
        for cs in metadata["core_shapes"]:
            sz = int(np.prod(cs))
            cores.append(_deserialize(data[pos : pos + sz * 4]).reshape(cs))
            pos += sz * 4
        C, U_inv, R = cores
        return (C.T @ U_inv @ R.T).T.reshape(shape).astype(np.float32)

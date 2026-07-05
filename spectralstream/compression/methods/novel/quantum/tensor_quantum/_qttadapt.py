from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class QTTAdapt:
    name = "qtt_adapt"
    category = "tensor_quantum"

    def _qtt_decompose(
        self, data_2d: np.ndarray, rank: int
    ) -> Tuple[list, int, int, int]:
        m, n = data_2d.shape[:2]
        dm = max(1, int(math.ceil(math.log2(m))))
        dn = max(1, int(math.ceil(math.log2(n))))
        pm = 1 << dm
        pn = 1 << dn
        tp = np.zeros((pm, pn), dtype=np.float64)
        tp[:m, :n] = data_2d
        cur = tp.reshape([2] * dm + [2] * dn)
        d = dm + dn
        cores = []
        chi = max(2, rank)
        prev_r = 1
        for k in range(d - 1):
            mode_dim = cur.shape[0] // prev_r
            unfolded = cur.reshape(prev_r * mode_dim, -1)
            U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            rk = min(chi, len(S))
            cores.append(U[:, :rk].reshape(prev_r, mode_dim, rk).copy())
            cur = S[:rk, None] * Vt[:rk, :]
            prev_r = rk
        cores.append(cur.reshape(prev_r, -1, 1).copy())
        return cores, dm, dn, d

    def compress(self, tensor: np.ndarray, rank: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            chi = max(2, rank)
            cores, dm, dn, d = self._qtt_decompose(t, chi)
            meta = dict(
                shape=orig_shape,
                dm=dm,
                dn=dn,
                d=d,
                core_shapes=[c.shape for c in cores],
                rank=chi,
            )
            data = struct.pack("<iii", dm, dn, d)
            for c in cores:
                data += _serialize(c.astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, dm=0, dn=0, d=0, core_shapes=[], rank=rank)
        data = struct.pack("<iii", 0, 0, 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def _qtt_reconstruct(self, cores, dm, dn, shape) -> np.ndarray:
        z = cores[0].astype(np.float64)
        for k in range(1, len(cores)):
            z = np.tensordot(z, cores[k].astype(np.float64), axes=([-1], [0]))
        if dn == 0:
            total = int(np.prod(shape))
            return z.ravel()[:total].reshape(shape)
        pm = 1 << dm
        pn = 1 << dn
        m, n = shape
        return z.reshape(pm, pn)[:m, :n]

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        dm, dn, d = struct.unpack_from("<iii", data, 0)
        if d == 0:
            flat = _deserialize(data[12:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 12
        cores = []
        for cs in metadata["core_shapes"]:
            sz = int(np.prod(cs))
            cores.append(_deserialize(data[pos : pos + sz * 4]).reshape(cs))
            pos += sz * 4
        return self._qtt_reconstruct(cores, dm, dn, shape).astype(np.float32)

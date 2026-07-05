from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


_BIG = 512 * 512


class DMRGSweep:
    """DMRG sweep via sequential SVD decomposition (MPS-style)."""

    name = "dmrg_sweep"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, bond_dim: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            chi = min(bond_dim, m, n)
            cores = []
            cur = t.copy()
            prev_r = 1
            n_split = max(2, min(m, 16))
            for k in range(n_split - 1):
                rows_per = max(1, m // n_split)
                if k < m % n_split:
                    rows_per += 1
                chunk_h = prev_r * rows_per
                if chunk_h * cur.shape[1] > cur.size:
                    chunk_h = cur.shape[0]
                unfolded = cur.reshape(chunk_h, -1)
                u, s, vt = np.linalg.svd(unfolded, full_matrices=False)
                rk = min(chi, len(s))
                cores.append(u[:, :rk].copy().astype(np.float32))
                cur = (s[:rk, None] * vt[:rk, :]).copy()
                prev_r = rk
            cores.append(cur.astype(np.float32))
            meta = dict(
                shape=orig_shape,
                bond_dim=chi,
                n_cores=len(cores),
                core_shapes=[c.shape for c in cores],
            )
            data = struct.pack("<ii", chi, len(cores))
            for c in cores:
                data += _serialize(c)
            return data, meta
        meta = dict(shape=orig_shape, bond_dim=bond_dim, n_cores=0, core_shapes=[])
        data = struct.pack("<ii", bond_dim, 0) + _serialize(
            t.ravel().astype(np.float32)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bond_dim, n_cores = struct.unpack_from("<ii", data, 0)
        if n_cores == 0:
            flat = _deserialize(data[8:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 8
        cores = []
        for cs in metadata["core_shapes"]:
            sz = int(np.prod(cs))
            cores.append(_deserialize(data[pos : pos + sz * 4]).reshape(cs))
            pos += sz * 4
        result = cores[0].astype(np.float64)
        for core in cores[1:]:
            result = result @ core
        return result.reshape(shape).astype(np.float32)

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class DensityMatrixRenorm:
    name = "density_matrix_renorm"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, bond_dim: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            chi = min(bond_dim, m, n)
            n_sites = min(m, chi)
            u, s, vt = np.linalg.svd(t, full_matrices=False)
            rk = min(chi, len(s))
            cores = [u[:, :rk].copy(), (s[:rk, None] * vt[:rk, :]).copy()]
            meta = dict(
                shape=orig_shape,
                bond_dim=chi,
                n_sites=n_sites,
                core_shapes=[c.shape for c in cores],
            )
            data = struct.pack("<ii", chi, n_sites)
            for c in cores:
                data += _serialize(c.astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, bond_dim=bond_dim, n_sites=0, core_shapes=[])
        data = struct.pack("<ii", bond_dim, 0) + _serialize(
            t.ravel().astype(np.float32)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bond_dim, n_sites = struct.unpack_from("<ii", data, 0)
        if n_sites == 0:
            flat = _deserialize(data[8:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 8
        m, n = shape
        cores = []
        for cs in metadata["core_shapes"]:
            sz = int(np.prod(cs))
            if sz == 0 or pos + sz * 4 > len(data):
                break
            cores.append(_deserialize(data[pos : pos + sz * 4]).reshape(cs))
            pos += sz * 4
        if len(cores) >= 2:
            return (cores[0].astype(np.float64) @ cores[1].astype(np.float64)).astype(
                np.float32
            )
        return np.zeros((m, n), dtype=np.float32)

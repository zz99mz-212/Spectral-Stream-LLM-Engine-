from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class TensorNetworkRegroup:
    name = "tensor_network_regroup"
    category = "tensor_quantum"

    def compress(
        self, tensor: np.ndarray, block_size: int = 2, chi: int = 8
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            bm = min(block_size, m)
            bn = min(block_size, n)
            isos = []
            for i in range(0, m, bm):
                for j in range(0, n, bn):
                    block = t[i : i + bm, j : j + bn]
                    if block.size == 0:
                        continue
                    u, s, vt = np.linalg.svd(block, full_matrices=False)
                    rk = min(chi, len(s))
                    isos.append(u[:, :rk].copy())
                    isos.append((s[:rk, None] * vt[:rk, :]).copy())
            if not isos:
                isos = [t.copy()]
            n_pairs = len(isos) // 2
            meta = dict(
                shape=orig_shape,
                block_size=bm,
                chi=chi,
                n_pairs=n_pairs,
                iso_shapes=[isos[k].shape for k in range(0, len(isos), 2)]
                if isos
                else [],
                dis_shapes=[isos[k + 1].shape for k in range(0, len(isos) - 1, 2)]
                if isos
                else [],
            )
            data = struct.pack("<iii", bm, chi, n_pairs)
            for arr in isos:
                data += _serialize(arr.astype(np.float32))
            return data, meta
        meta = dict(
            shape=orig_shape,
            block_size=block_size,
            chi=chi,
            n_pairs=0,
            iso_shapes=[],
            dis_shapes=[],
        )
        data = struct.pack("<iii", block_size, chi, 0) + _serialize(
            t.ravel().astype(np.float32)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bm, chi, n_pairs = struct.unpack_from("<iii", data, 0)
        if n_pairs == 0:
            flat = _deserialize(data[12:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 12
        m, n = shape
        recon = np.zeros((m, n), dtype=np.float64)
        iso_shapes = metadata["iso_shapes"]
        dis_shapes = metadata["dis_shapes"]
        for pi in range(min(n_pairs, len(iso_shapes), len(dis_shapes))):
            iso_sz = int(np.prod(iso_shapes[pi]))
            dis_sz = int(np.prod(dis_shapes[pi]))
            iso = _deserialize(data[pos : pos + iso_sz * 4]).reshape(iso_shapes[pi])
            pos += iso_sz * 4
            dis = _deserialize(data[pos : pos + dis_sz * 4]).reshape(dis_shapes[pi])
            pos += dis_sz * 4
            i = (pi // max(1, n // bm)) * bm
            j = (pi % max(1, n // bm)) * bm
            if i < m and j < n:
                block = iso[:, :chi] @ dis[:chi, :]
                hi = min(bm, m - i)
                wj = min(dis.shape[1], n - j)
                recon[i : i + hi, j : j + wj] = block[:hi, :wj]
        return recon.astype(np.float32)

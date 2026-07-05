from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class TopologicalOrder:
    name = "topological_order"
    category = "tensor_quantum"

    def compress(
        self, tensor: np.ndarray, n_plaquettes: int = None
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            if n_plaquettes is None:
                n_plaquettes = max(1, min(16, m * n // 4))
            plaq_size = max(2, int(math.sqrt(m * n / max(n_plaquettes, 1))))
            plaq_rows = max(1, m // plaq_size)
            plaq_cols = max(1, n // plaq_size)
            constraints = []
            for i in range(plaq_rows):
                for j in range(plaq_cols):
                    bi = i * plaq_size
                    bj = j * plaq_size
                    plaq = t[bi : min(bi + plaq_size, m), bj : min(bj + plaq_size, n)]
                    constraints.append(float(np.mean(plaq)))
            meta = dict(
                shape=orig_shape,
                n_plaquettes=len(constraints),
                plaq_size=plaq_size,
                plaq_rows=plaq_rows,
                plaq_cols=plaq_cols,
            )
            data = struct.pack(
                "<iiii", len(constraints), plaq_size, plaq_rows, plaq_cols
            )
            data += _serialize(np.array(constraints, dtype=np.float32))
            return data, meta
        meta = dict(
            shape=orig_shape, n_plaquettes=0, plaq_size=0, plaq_rows=0, plaq_cols=0
        )
        data = struct.pack("<iiii", 0, 0, 0, 0) + _serialize(
            t.ravel().astype(np.float32)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_plaq, plaq_size, plaq_rows, plaq_cols = struct.unpack_from("<iiii", data, 0)
        if n_plaq == 0:
            flat = _deserialize(data[16:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 16
        constraints = _deserialize(data[pos : pos + n_plaq * 4])
        m, n = shape
        recon = np.zeros((m, n), dtype=np.float64)
        idx = 0
        for i in range(plaq_rows):
            for j in range(plaq_cols):
                if idx >= len(constraints):
                    break
                bi = i * plaq_size
                bj = j * plaq_size
                h = min(plaq_size, m - bi)
                w = min(plaq_size, n - bj)
                recon[bi : bi + h, bj : bj + w] = constraints[idx]
                idx += 1
        return recon.astype(np.float32)

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class MultiStability:
    """Attractor switching between A_1...A_k attractor states."""

    name = "multi_stability"
    category = "novel_chaos"

    def compress(self, tensor: np.ndarray, n_attractors: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape

        block_size = max(1, n // n_attractors)
        attractors = []
        for i in range(n_attractors):
            start = i * block_size
            end = min((i + 1) * block_size, n)
            if start < end:
                attractors.append(t[:, start:end])

        attractor_params = []
        for A in attractors:
            UA, SA, VtA = np.linalg.svd(A, full_matrices=False)
            k = max(1, min(4, len(SA)))
            attractor_params.append(
                {
                    "U": UA[:, :k].astype(np.float32),
                    "S": SA[:k].astype(np.float32),
                    "Vt": VtA[:k, :].astype(np.float32),
                }
            )

        boundaries = np.linspace(0, n, n_attractors + 1).astype(int)

        data = b""
        block_sizes = []
        for i, A in enumerate(attractors):
            _, n_A = A.shape
            block_sizes.append(n_A)
            UA, SA, VtA = np.linalg.svd(A, full_matrices=False)
            k = max(1, min(4, len(SA)))
            data += struct.pack("<ii", k, n_A)
            data += _serialize(UA[:, :k].astype(np.float32))
            data += _serialize(SA[:k].astype(np.float32))
            data += _serialize(VtA[:k, :].astype(np.float32))

        data += _serialize(boundaries.astype(np.int32))

        meta = dict(
            shape=tensor.shape,
            n_attractors=n_attractors,
            block_size=block_size,
            block_sizes=block_sizes,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_attractors = metadata["n_attractors"]
        m, n = shape

        recon = np.zeros((m, n), dtype=np.float64)
        pos = 0
        for i in range(n_attractors):
            header = data[pos : pos + 8]
            pos += 8
            if len(header) < 8:
                break
            k, n_A = struct.unpack("<ii", header)

            U = _deserialize(data[pos : pos + m * k * 4]).reshape(m, k)
            pos += m * k * 4
            S = _deserialize(data[pos : pos + k * 4])
            pos += k * 4
            Vt = _deserialize(data[pos : pos + k * n_A * 4]).reshape(k, n_A)
            pos += k * n_A * 4

            block_recon = (U * S) @ Vt
            start = i * (n // n_attractors)
            end = start + n_A
            recon[:, start:end] = block_recon

        return recon.reshape(shape).astype(np.float32)

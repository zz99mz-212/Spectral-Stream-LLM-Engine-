from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class HyperspectralTensor:
    name = "hyperspectral_tensor"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            grad = np.zeros_like(t)
            grad[:, :-1] = t[:, 1:] - t[:, :-1]
            views = np.stack([t, t.T, grad], axis=-1)
            r1 = min(rank, m)
            r2 = min(rank, n)
            r3 = min(rank, 3)
            u1, _, _ = np.linalg.svd(views.reshape(m, n * 3), full_matrices=False)
            u1 = u1[:, :r1]
            u2, _, _ = np.linalg.svd(
                views.transpose(1, 0, 2).reshape(n, m * 3), full_matrices=False
            )
            u2 = u2[:, :r2]
            u3, _, _ = np.linalg.svd(
                views.transpose(2, 0, 1).reshape(3, m * n), full_matrices=False
            )
            u3 = u3[:, :r3]
            core = np.tensordot(views, u1.T, axes=([0], [1]))
            core = np.tensordot(core, u2.T, axes=([0], [1]))
            core = np.tensordot(core, u3.T, axes=([0], [1]))
            meta = dict(
                shape=orig_shape, r1=r1, r2=r2, r3=r3, core_shape=list(core.shape)
            )
            data = struct.pack("<iii", r1, r2, r3)
            data += _serialize(u1.astype(np.float32))
            data += _serialize(u2.astype(np.float32))
            data += _serialize(u3.astype(np.float32))
            data += _serialize(core.astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, r1=0, r2=0, r3=0, core_shape=[])
        data = struct.pack("<iii", 0, 0, 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        r1, r2, r3 = struct.unpack_from("<iii", data, 0)
        if r1 == 0:
            flat = _deserialize(data[12:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 12
        m, n = shape
        u1 = _deserialize(data[pos : pos + m * r1 * 4]).reshape(m, r1)
        pos += m * r1 * 4
        u2 = _deserialize(data[pos : pos + n * r2 * 4]).reshape(n, r2)
        pos += n * r2 * 4
        u3 = _deserialize(data[pos : pos + 3 * r3 * 4]).reshape(3, r3)
        pos += 3 * r3 * 4
        cs = metadata["core_shape"]
        sz = int(np.prod(cs))
        core = _deserialize(data[pos : pos + sz * 4]).reshape(cs)
        recon = np.tensordot(core, u1, axes=([0], [1]))
        recon = np.tensordot(recon, u2, axes=([0], [1]))
        recon = np.tensordot(recon, u3, axes=([0], [1]))
        return np.real(recon[:, :, 0].astype(np.float32))

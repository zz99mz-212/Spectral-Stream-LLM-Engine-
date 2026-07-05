from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class MBQCCompress:
    name = "mbqc_compress"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, n_angles: int = None) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            if n_angles is None:
                n_angles = min(m, 4)
            n_angles = max(2, min(n_angles, m))
            angles = np.arctan2(t[:n_angles, :n_angles].ravel(), 1.0)
            ent = np.full(len(angles) - 1, np.pi / 4, dtype=np.float64)
            scale = float(np.linalg.norm(t))
            meta = dict(shape=orig_shape, n_angles=len(angles), scale=scale)
            data = struct.pack("<if", len(angles), scale)
            data += _serialize(angles.astype(np.float32))
            data += _serialize(ent.astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, n_angles=0, scale=0.0)
        data = struct.pack("<if", 0, 0.0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_angles, scale = struct.unpack_from("<if", data, 0)
        pos = struct.calcsize("<if")
        if n_angles == 0:
            flat = _deserialize(data[pos:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        angles = _deserialize(data[pos : pos + n_angles * 4])
        pos += n_angles * 4
        ent = _deserialize(data[pos : pos + (n_angles - 1) * 4])
        corr = np.cos(angles)
        side = int(math.sqrt(n_angles))
        if side * side < n_angles:
            side += 1
        corr_mat = np.outer(corr[:side], corr[:side]) * (scale / max(1e-10, side))
        m, n = shape
        recon = np.zeros((m, n), dtype=np.float64)
        h, w = min(side, m), min(side, n)
        recon[:h, :w] = corr_mat[:h, :w]
        return recon.astype(np.float32)

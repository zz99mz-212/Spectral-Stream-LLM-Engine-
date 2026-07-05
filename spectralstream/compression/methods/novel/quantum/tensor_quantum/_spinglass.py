from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class SpinGlass:
    name = "spin_glass"
    category = "tensor_quantum"

    def compress(
        self, tensor: np.ndarray, keep_frac: float = 0.3
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            J_flat = t.ravel()
            k = max(1, min(len(J_flat), int(keep_frac * len(J_flat))))
            thr = np.sort(np.abs(J_flat))[-k]
            mask = np.abs(J_flat) >= thr
            idx = np.where(mask)[0]
            vals = J_flat[mask]
            meta = dict(shape=orig_shape, keep_frac=keep_frac, n_couplings=len(idx))
            data = struct.pack("<fi", float(keep_frac), len(idx))
            data += _serialize(idx.astype(np.int32))
            data += vals.astype(np.float16).tobytes()
            return data, meta
        meta = dict(shape=orig_shape, keep_frac=0.0, n_couplings=0)
        data = struct.pack("<fi", 0.0, 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        keep_frac, n_coup = struct.unpack_from("<fi", data, 0)
        pos = struct.calcsize("<fi")
        if n_coup == 0:
            flat = _deserialize(data[pos:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        idx = _deserialize(data[pos : pos + n_coup * 4]).astype(int)
        pos += n_coup * 4
        vals = np.frombuffer(data[pos:], dtype=np.float16).astype(np.float64)
        flat = np.zeros(int(np.prod(shape)), dtype=np.float64)
        idxi = idx[idx < len(flat)]
        flat[idxi] = vals[: len(idxi)]
        return flat.reshape(shape).astype(np.float32)

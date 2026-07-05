from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class FloquetTensor:
    name = "floquet_tensor"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, n_freqs: int = 8) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            freq_domain = np.fft.fftn(t)
            flat = freq_domain.ravel()
            k = min(n_freqs * t.shape[0], len(flat) // 2)
            k = max(1, k)
            idx = np.argpartition(np.abs(flat), -k)[-k:]
            idx = idx[np.argsort(-np.abs(flat[idx]))]
            vals = flat[idx]
            meta = dict(shape=orig_shape, n_freqs=n_freqs, n_components=len(idx))
            data = (
                struct.pack("<i", len(idx))
                + _serialize(idx.astype(np.int32))
                + vals.astype(np.complex64).tobytes()
            )
            return data, meta
        meta = dict(shape=orig_shape, n_freqs=n_freqs, n_components=0)
        data = struct.pack("<i", 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_components = struct.unpack_from("<i", data, 0)[0]
        if n_components == 0:
            flat = _deserialize(data[4:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        pos = 4
        idx = _deserialize(data[pos : pos + n_components * 4]).astype(int)
        pos += n_components * 4
        vals = np.frombuffer(data[pos:], dtype=np.complex64).astype(np.complex128)
        n = int(np.prod(shape))
        coeffs = np.zeros(n, dtype=np.complex128)
        valid = idx < n
        coeffs[idx[valid]] = vals[valid]
        return np.real(np.fft.ifftn(coeffs.reshape(shape))).astype(np.float32)

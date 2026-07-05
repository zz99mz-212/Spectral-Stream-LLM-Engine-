from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np
from ._qttadapt import QTTAdapt


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class QTTFourier:
    name = "qtt_fourier"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, rank: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            coeffs = np.abs(np.fft.fft2(t))
            chi = max(2, rank)
            qtt = QTTAdapt()
            cores, dm, dn, d = qtt._qtt_decompose(coeffs, chi)
            meta = dict(
                shape=orig_shape,
                dm=dm,
                dn=dn,
                d=d,
                rank=chi,
                core_shapes=[c.shape for c in cores],
            )
            data = struct.pack("<iii", dm, dn, d)
            for c in cores:
                data += _serialize(c.astype(np.float32))
            return data, meta
        meta = dict(shape=orig_shape, dm=0, dn=0, d=0, rank=rank, core_shapes=[])
        data = struct.pack("<iii", 0, 0, 0) + _serialize(t.ravel().astype(np.float32))
        return data, meta

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
        qtt = QTTAdapt()
        mag = qtt._qtt_reconstruct(cores, dm, dn, shape)
        return np.real(np.fft.ifft2(mag * np.exp(1j * np.zeros_like(mag)))).astype(
            np.float32
        )

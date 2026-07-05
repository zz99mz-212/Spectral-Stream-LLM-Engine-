from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumAmplitude:
    name = "quantum_amplitude"
    category = "tensor_quantum"

    def compress(self, tensor: np.ndarray, bond_dim: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        flat = t.ravel()
        norm = np.linalg.norm(flat) + 1e-30
        psi = flat / norm
        n = len(psi)
        d = max(1, int(math.ceil(math.log2(n))))
        pad = (1 << d) - n
        if pad > 0:
            psi = np.pad(psi, (0, pad))
        chi = max(2, bond_dim)
        qtt = QTTAdapt()
        cores, dm, _, _ = qtt._qtt_decompose(psi.reshape(-1, 1), chi)
        meta = dict(
            shape=orig_shape,
            d=d,
            norm=float(norm),
            core_shapes=[c.shape for c in cores],
            bond_dim=chi,
        )
        data = struct.pack("<id", d, norm)
        for c in cores:
            data += _serialize(c.astype(np.float32))
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        d, norm = struct.unpack_from("<id", data, 0)
        pos = struct.calcsize("<id")
        cores = []
        for cs in metadata["core_shapes"]:
            sz = int(np.prod(cs))
            cores.append(_deserialize(data[pos : pos + sz * 4]).reshape(cs))
            pos += sz * 4
        qtt = QTTAdapt()
        total = int(np.prod(shape))
        recon_1d = qtt._qtt_reconstruct(cores, d, 0, (total, 1)).ravel()
        return (recon_1d[:total] * norm).reshape(shape).astype(np.float32)

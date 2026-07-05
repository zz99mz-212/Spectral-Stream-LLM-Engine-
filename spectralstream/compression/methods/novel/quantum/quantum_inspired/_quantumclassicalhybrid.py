from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class QuantumClassicalHybrid:
    """Variational quantum-classical: quantum for subspace, classical for residual.
    Quantum part captures coherent structure; classical handles fine details.
    """

    name = "quantum_classical_hybrid"
    category = "quantum_compression"

    def compress(
        self,
        tensor: np.ndarray,
        subspace_rank: int = 4,
        block_size: int = 64,
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        m, n = t.shape
        k = min(subspace_rank, m, n)
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        quantum_part = U_k @ np.diag(S_k) @ Vt_k
        residual = t - quantum_part
        residual_flat = residual.ravel()
        res_n = len(residual_flat)
        padded_n = int(math.ceil(res_n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float64)
        padded[:res_n] = residual_flat
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
        quantized = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)
        meta = dict(
            shape=orig_shape,
            m=m,
            n=n,
            k=k,
            block_size=block_size,
            res_n=res_n,
        )
        data = struct.pack("<IIII", m, n, k, block_size)
        data += _serialize(U_k.astype(np.float32))
        data += _serialize(S_k.astype(np.float32))
        data += _serialize(Vt_k.astype(np.float32))
        data += scales.astype(np.float32).tobytes()
        data += quantized.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n, k, block_size = struct.unpack_from("<IIII", data, 0)
        pos = 16
        U_k = _deserialize(data[pos : pos + m * k * 4]).reshape(m, k)
        pos += m * k * 4
        S_k = _deserialize(data[pos : pos + k * 4])
        pos += k * 4
        Vt_k = _deserialize(data[pos : pos + k * n * 4]).reshape(k, n)
        pos += k * n * 4
        res_n = metadata["res_n"]
        n_blocks = int(math.ceil(res_n / block_size))
        scales = np.frombuffer(
            data[pos : pos + n_blocks * 4], dtype=np.float32
        ).reshape(-1, 1)
        pos += n_blocks * 4
        quantized = (
            np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
            .reshape(-1, block_size)
            .astype(np.float32)
        )
        quantum = U_k @ np.diag(S_k) @ Vt_k
        residual = (quantized * scales).ravel()[:res_n]
        flat = quantum.ravel() + residual
        return flat.reshape(shape).astype(np.float32)

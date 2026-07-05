from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class QuantumErrorCorrecting:
    name = "quantum_error_correcting"
    category = "tensor_quantum"

    def compress(
        self, tensor: np.ndarray, code_rate: float = 0.5
    ) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        orig_shape = t.shape
        if t.ndim == 2:
            m, n = t.shape
            k = max(1, min(m, n, int(n * code_rate)))
            u, s, vt = np.linalg.svd(t, full_matrices=False)
            rank = min(k, len(s))
            approx = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
            syndrome = t - approx
            mask = np.abs(syndrome) > np.percentile(np.abs(syndrome), 80)
            sy_idx = np.argwhere(mask)
            sy_vals = syndrome[mask]
            meta = dict(
                shape=orig_shape,
                code_rate=code_rate,
                rank=rank,
                n_syndromes=len(sy_idx),
            )
            data = struct.pack("<ffi", code_rate, float(rank), len(sy_idx))
            data += _serialize(u[:, :rank].astype(np.float32))
            data += _serialize(s[:rank].astype(np.float32))
            data += _serialize(vt[:rank, :].astype(np.float32))
            data += _serialize(sy_idx.astype(np.int16))
            data += sy_vals.astype(np.float16).tobytes()
            return data, meta
        meta = dict(shape=orig_shape, code_rate=0.0, rank=0, n_syndromes=0)
        data = struct.pack("<ffi", 0.0, 0.0, 0) + _serialize(
            t.ravel().astype(np.float32)
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        code_rate, rank_f, n_syn = struct.unpack_from("<ffi", data, 0)
        rank = int(rank_f)
        pos = struct.calcsize("<ffi")
        if rank == 0:
            flat = _deserialize(data[pos:])
            return flat[: int(np.prod(shape))].reshape(shape).astype(np.float32)
        m, n = shape
        u = _deserialize(data[pos : pos + m * rank * 4]).reshape(m, rank)
        pos += m * rank * 4
        s = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        vt = _deserialize(data[pos : pos + rank * n * 4]).reshape(rank, n)
        pos += rank * n * 4
        recon = (u * s) @ vt
        if n_syn > 0:
            n_int16 = n_syn * 2
            sy_idx = np.frombuffer(
                data[pos : pos + n_int16 * 2], dtype=np.int16
            ).reshape(-1, 2)
            pos += n_int16 * 2
            sy_vals = np.frombuffer(
                data[pos : pos + n_syn * 2], dtype=np.float16
            ).astype(np.float64)
            valid = (sy_idx[:, 0] < m) & (sy_idx[:, 1] < n)
            recon[sy_idx[valid, 0], sy_idx[valid, 1]] += sy_vals[valid]
        return recon.astype(np.float32)

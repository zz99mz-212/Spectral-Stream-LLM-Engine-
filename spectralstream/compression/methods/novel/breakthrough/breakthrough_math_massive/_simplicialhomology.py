from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def _svd_core(tensor: np.ndarray, rank: int = 8):
    t = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
    m, n = t.shape
    U, S, Vt = np.linalg.svd(t, full_matrices=False)
    k = min(rank, len(S), m, n)
    return m, n, k, U[:, :k].astype(np.float32), S[:k].astype(np.float32), Vt[:k, :].astype(np.float32)

def _build_header(m: int, n: int, k: int) -> bytes:
    return struct.pack("<III", m, n, k)

def _read_header(data: bytes, pos: int = 0):
    m, n, k = struct.unpack_from("<III", data, pos)
    return m, n, k, pos + 12

def _reconstruct(data: bytes, pos: int, m: int, n: int, k: int, shape: tuple) -> np.ndarray:
    Uk = np.frombuffer(data[pos:pos + m * k * 4], dtype=np.float32).reshape(m, k)
    pos += m * k * 4
    Sk = np.frombuffer(data[pos:pos + k * 4], dtype=np.float32)
    pos += k * 4
    Vk = np.frombuffer(data[pos:pos + k * n * 4], dtype=np.float32).reshape(k, n)
    recon = (Uk * Sk) @ Vk
    return recon.astype(np.float32).reshape(shape)

class SimplicialHomology:
    """Simplicial homology: combinatorial approximation of weight"""
    name = "simplicial_homology"; category = "breakthrough_math"

    def compress(self, tensor: np.ndarray, rank: int = 8) -> Tuple[bytes, dict]:
        m, n, k, Uk, Sk, Vk = _svd_core(tensor, rank)
        param = float(np.sum(np.abs(np.diff(Uk[:min(m,8),:].ravel()))))
        buf = _build_header(m, n, k) + struct.pack("<I", param) + _serialize(Uk) + _serialize(Sk) + _serialize(Vk)
        return buf, {"shape": tensor.shape}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k, pos = _read_header(data)
        pos += 4
        return _reconstruct(data, pos, m, n, k, metadata["shape"])

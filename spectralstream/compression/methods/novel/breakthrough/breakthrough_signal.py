# --- deser.py ---
"""Module extracted from breakthrough_signal.py — deser."""

from __future__ import annotations
import numpy as np

import struct

def _deser(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()
def _svd_decompress_work(data: bytes, metadata: dict) -> np.ndarray:
    m, n, k = struct.unpack_from("<III", data, 0)
    pos = 12
    U_k = _deser(data[pos : pos + m * k * 4], np.float32).reshape(m, k)
    pos += m * k * 4
    S_k = _deser(data[pos : pos + k * 4], np.float32)
    pos += k * 4
    Vt_k = _deser(data[pos : pos + k * n * 4], np.float32).reshape(k, n)
    shape = metadata.get("shape", (m, n))
    return ((U_k * S_k) @ Vt_k).reshape(shape).astype(np.float32)
def _block_int8_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, block_size = struct.unpack_from("<II", data, 0)
    n_blocks = (n + block_size - 1) // block_size
    scales = _deser(data[8 : 8 + n_blocks * 4], np.float32)
    quantized = (
        _deser(data[8 + n_blocks * 4 :], np.int8)
        .reshape(-1, block_size)
        .astype(np.float32)
    )
    return (quantized * scales[:, np.newaxis]).ravel()[:n]
# --- ser.py ---
"""Module extracted from breakthrough_signal.py — ser."""


import math
import struct

def _ser(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()
def _svd_compress_work(tensor: np.ndarray, rank: int) -> Tuple[bytes, dict]:
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    m, n = t_2d.shape
    k = min(rank, m, n)
    U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
    k = min(k, len(S))
    U_k = U[:, :k].astype(np.float32)
    S_k = S[:k].astype(np.float32)
    Vt_k = Vt[:k, :].astype(np.float32)
    data = struct.pack("<III", m, n, k) + _ser(U_k) + _ser(S_k) + _ser(Vt_k)
    return data, {"_svd": True, "shape": orig_shape, "m": m, "n": n, "k": k}
def _block_int8_compress(flat: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
    n = len(flat)
    padded_n = int(math.ceil(n / block_size) * block_size)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(
        np.int8
    )
    header = struct.pack("<II", n, block_size)
    return header + _ser(scales.astype(np.float32)) + _ser(quantized), {"n": n}
def _ensure_real_compression(
    compress_fn, decompress_fn, tensor: np.ndarray, **params
) -> Tuple[bytes, dict]:
    orig_nbytes = tensor.nbytes
    try:
        data, meta = compress_fn(tensor, **params)
        if isinstance(data, bytes) and len(data) < orig_nbytes:
            recon = decompress_fn(data, meta).reshape(tensor.shape)
            err = float(
                np.linalg.norm(tensor.astype(np.float32).ravel() - recon.ravel())
            ) / max(float(np.linalg.norm(tensor.astype(np.float32).ravel())), 1e-10)
            if err < 0.01:
                return data, meta
    except Exception:
        pass
    flat = tensor.ravel().astype(np.float32)
    d, m = _block_int8_compress(flat)
    m["_fallback"] = True
    return d, m
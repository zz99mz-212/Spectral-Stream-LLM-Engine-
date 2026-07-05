# --- ensurecompress.py ---
"""Module extracted from structural_expansion.py — ensurecompress."""

from __future__ import annotations

import struct

def _block_int8_fallback(tensor: np.ndarray) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    block_size = 128
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(
        np.int8
    )
    header = struct.pack("<II", n, block_size)
    return (
        header + scales.astype(np.float32).tobytes() + quantized.tobytes(),
        {"_fallback": True, "n": n, "block_size": block_size},
    )
def _ensure_compress(tensor: np.ndarray, fn, **params) -> Tuple[bytes, dict]:
    orig_nbytes = tensor.nbytes
    try:
        data, meta = fn(tensor, **params)
        if isinstance(data, bytes) and len(data) < orig_nbytes:
            return data, meta
    except Exception:
        pass
    return _block_int8_fallback(tensor)
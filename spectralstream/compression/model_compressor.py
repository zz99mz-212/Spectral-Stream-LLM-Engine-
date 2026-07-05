# --- zigzagscan.py ---
"""Module extracted from model_compressor.py — zigzagscan."""

from __future__ import annotations


import re
from spectralstream.format.sscx_format import (
    SSCXWriter,
    SSCXReader,
    SSCXHeader,
    SSCXTensorEntry as SSCXTensorInfo,
    SSCXLayerEntry as SSCXLayerInfo,
    SSCXFooter,
    COMP_RAW,
    COMP_DCT,
    COMP_SPECTRAL,
    COMP_INT8,
    COMP_INT4,
    COMP_DELTA,
    COMP_NAMES,
    DTYPE_FP32,
    DTYPE_FP16,
    DTYPE_INT8,
    _align_up,
    _format_size,
    _crc32,
)
from spectralstream.core.math_primitives import dct_2d, idct_2d
import struct

def _zigzag_scan(matrix: np.ndarray) -> np.ndarray:
    """Zigzag scan of 2D matrix to 1D (JPEG-style)."""
    n, m = matrix.shape
    result = []
    for d in range(n + m - 1):
        if d % 2 == 0:
            for i in range(max(0, d - m + 1), min(d + 1, n)):
                j = d - i
                if j < m:
                    result.append(matrix[i, j])
        else:
            for i in range(min(d + 1, n) - 1, max(0, d - m + 1) - 1, -1):
                j = d - i
                if 0 <= j < m:
                    result.append(matrix[i, j])
    return np.array(result, dtype=np.float32)
def _compress_dct_block(tensor: np.ndarray, block_size: int = 64) -> tuple[bytes, int]:
    """Compress 2D tensor using block-wise DCT with coefficient pruning."""
    orig_size = tensor.nbytes
    mat = tensor.astype(np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    m, n = mat.shape
    # Use smaller blocks for small matrices to reduce overhead
    if max(m, n) <= 64:
        block_size = min(block_size, 16)

    buf = struct.pack("<IHI", m, n, block_size)

    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            bh = min(block_size, m - i)
            bw = min(block_size, n - j)
            block = np.zeros((block_size, block_size), dtype=np.float32)
            block[:bh, :bw] = mat[i : i + bh, j : j + bw]

            coeffs = dct_2d(block.astype(np.float64)).astype(np.float32)
            flat = _zigzag_scan(coeffs)
            n_keep = max(1, int(len(flat) * 0.40))
            threshold = np.sort(np.abs(flat))[::-1][min(n_keep - 1, len(flat) - 1)]
            mask = np.abs(flat) >= threshold
            kept = flat[mask]
            indices = np.where(mask)[0].astype(np.int32)

            amax = float(np.max(np.abs(kept))) if len(kept) > 0 else 1.0
            scale = amax / 127.0 if amax > 1e-8 else 1.0
            quantized = np.clip(np.round(kept / scale), -128, 127).astype(np.int8)

            buf += struct.pack("<HHH", bh, bw, len(indices))
            buf += struct.pack("<f", scale)
            buf += indices.tobytes()
            buf += quantized.tobytes()

    return bytes(buf), orig_size
# --- zigzagunscan.py ---
"""Module extracted from model_compressor.py — zigzagunscan."""


import struct

def _zigzag_unscan(flat: np.ndarray, size: int) -> np.ndarray:
    """Inverse zigzag scan: 1D back to 2D square matrix."""
    matrix = np.zeros((size, size), dtype=np.float32)
    idx = 0
    for d in range(2 * size - 1):
        if d % 2 == 0:
            for i in range(max(0, d - size + 1), min(d + 1, size)):
                j = d - i
                if j < size and idx < len(flat):
                    matrix[i, j] = flat[idx]
                    idx += 1
        else:
            for i in range(min(d + 1, size) - 1, max(0, d - size + 1) - 1, -1):
                j = d - i
                if 0 <= j < size and idx < len(flat):
                    matrix[i, j] = flat[idx]
                    idx += 1
    return matrix
def _decompress_dct_block(
    data: bytes, shape: tuple, block_size: int = 64
) -> np.ndarray:
    """Decompress block-wise DCT compressed tensor."""
    pos = 0
    m, n, bs = struct.unpack_from("<IHI", data, pos)
    pos += 10
    out = np.zeros((m, n), dtype=np.float32)

    for i in range(0, m, bs):
        for j in range(0, n, bs):
            if pos + 6 > len(data):
                break
            bh, bw, n_coeffs = struct.unpack_from("<HHH", data, pos)
            pos += 6
            scale = struct.unpack_from("<f", data, pos)[0]
            pos += 4
            indices = np.frombuffer(data[pos : pos + n_coeffs * 4], dtype=np.int32)
            pos += n_coeffs * 4
            quantized = np.frombuffer(data[pos : pos + n_coeffs], dtype=np.int8)
            pos += n_coeffs

            flat = np.zeros(bs * bs, dtype=np.float32)
            vals = quantized.astype(np.float32) * scale
            flat[indices] = vals
            coeffs = _zigzag_unscan(flat, bs)
            block = idct_2d(coeffs.astype(np.float64)).astype(np.float32)
            out[i : i + bh, j : j + bw] = block[:bh, :bw]

    target_h, target_w = shape[0], shape[1]
    if out.shape[0] >= target_h and out.shape[1] >= target_w:
        return out[:target_h, :target_w]
    return out
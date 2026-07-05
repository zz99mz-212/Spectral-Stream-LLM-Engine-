from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.novel._common import (
    _block_int8_fallback,
    _block_int8_decompress,
    _block_int4_compress,
    _block_int4_decompress,
    _nf4_compress,
    _nf4_decompress,
    _svd_compress,
    _svd_decompress,
    _svd_then_quant,
    _svd_then_dequant,
    _dct_then_quant,
    _dct_then_dequant,
    _fourier_then_quant,
    _fourier_then_dequant,
    _hadamard_then_quant,
    _hadamard_then_dequant,
    _sparsify_2of4,
    _sparsify_2of4_decompress,
    _sparsify_block,
    _sparsify_block_decompress,
    _unstructured_prune,
    _unstructured_prune_decompress,
    _product_quantize,
    _product_quantize_decompress,
    _tt_compress,
    _tt_decompress,
    _cp_compress,
    _cp_decompress,
    _tucker_compress,
    _tucker_decompress,
    _kronecker_compress,
    _kronecker_decompress,
    _huffman_encode,
    _huffman_decode,
    _zstd_compress,
    _zstd_decompress,
)


def _as_method(fn):
    """Wrap a bare (data, meta) function so it accepts self."""
    return lambda self, data, meta=None: fn(data, meta or {})

class Fourier_ThenSparse:
    name = "fourier_then_sparse"
    category = "breakthrough_hybrid"

    def compress(self, tensor, **params):
        flat = tensor.ravel().astype(np.float32)
        fft_coeffs = np.fft.fft(flat).real.astype(np.float32)
        threshold = params.get("threshold", 0.01 * np.max(np.abs(fft_coeffs)))
        mask = np.abs(fft_coeffs) > threshold
        kept = fft_coeffs[mask]
        indices = np.where(mask)[0].astype(np.uint32)
        header = struct.pack("<II", len(flat), len(kept))
        return header + kept.tobytes() + indices.tobytes(), {
            "_fourier_sparse": True,
            "n": len(flat),
        }

    def decompress(self, data, meta=None):
        meta = meta or {}
        n, n_kept = struct.unpack_from("<II", data, 0)
        pos = 8
        kept = np.frombuffer(data[pos : pos + n_kept * 4], dtype=np.float32)
        pos += n_kept * 4
        indices = np.frombuffer(data[pos : pos + n_kept * 4], dtype=np.uint32)
        coeffs = np.zeros(n, dtype=np.complex128)
        for i, idx in enumerate(indices):
            if idx < n:
                coeffs[idx] = kept[i]
        return np.fft.ifft(coeffs).real.astype(np.float32)

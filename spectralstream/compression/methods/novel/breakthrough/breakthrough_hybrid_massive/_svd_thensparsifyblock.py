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

class SVD_ThenSparsifyBlock:
    name = "svd_then_sparsify_block"
    category = "breakthrough_hybrid"

    def compress(self, tensor, **params):
        return _sparsify_block(
            tensor, params.get("block_size", 32), params.get("sparsity", 0.5)
        )

    def decompress(self, data, meta=None):
        return _sparsify_block_decompress(data, meta or {})

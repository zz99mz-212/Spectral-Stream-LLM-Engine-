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
    _sparsify_2of4,
    _sparsify_2of4_decompress,
    _sparsify_block,
    _sparsify_block_decompress,
    _unstructured_prune,
    _unstructured_prune_decompress,
    _product_quantize,
    _product_quantize_decompress,
    _svd_compress,
    _svd_decompress,
    _ensure_compression,
)
from spectralstream.compression.methods.novel.breakthrough.breakthrough_info_massive import (
    _uniform_quantize_compress,
    _uniform_quantize_decompress,
    _log_quantize_compress,
    _log_quantize_decompress,
    _dpcm_encode,
    _dpcm_decode,
)


class MultiTerminalEntropy:
    name = "multiterminalentropy"
    category = "information_theory_2"

    def compress(self, tensor, **params):
        return _uniform_quantize_compress(tensor, 6)

    decompress = staticmethod(_uniform_quantize_decompress)

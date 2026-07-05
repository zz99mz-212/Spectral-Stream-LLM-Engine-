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
    _svd_compress,
    _svd_decompress,
    _sparsify_2of4,
    _sparsify_2of4_decompress,
    _nf4_compress,
    _nf4_decompress,
)


def _topo_compress(tensor, method="topo_hadamard_svd", rank=0):
    k = rank if rank > 0 else max(1, min(tensor.shape[0] if tensor.ndim >= 2 else 1, tensor.size // max(1, tensor.shape[0] if tensor.ndim >= 2 else 1)) // 10)
    k = min(k, 64)
    return _svd_compress(tensor, k)

def _topo_decompress(data, meta):
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class MinimalModel:
    """Topological compression: m i n i m a l m o d e l — strategy #4"""

    name = "minimalmodel"
    category = "revolutionary_topological"

    def compress(self, tensor, **params):
        return _topo_compress(tensor, "svd", params.get("rank", 0))

    decompress = staticmethod(_topo_decompress)

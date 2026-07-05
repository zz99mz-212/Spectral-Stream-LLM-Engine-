from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.novel._common import (
    _block_int8_fallback,
    _block_int8_decompress,
    _svd_compress,
    _svd_decompress,
)


def _topo_compress(tensor, method="svd", rank=0):
    return _svd_compress(tensor, rank)

def _topo_decompress(data, meta):
    return _svd_decompress(data, meta)

class ClassifyingSpace:
    """Topological compression: Classifying space"""

    name = "classifyingspace"
    category = "revolutionary_topological"

    def compress(self, tensor, **params):
        return _topo_compress(tensor, "svd", params.get("rank", 0))

    decompress = staticmethod(_topo_decompress)

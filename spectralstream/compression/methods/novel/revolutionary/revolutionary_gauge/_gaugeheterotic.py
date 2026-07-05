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


def _gauge_compress(tensor, method="svd", rank=0):
    if method == "svd":
        return _svd_compress(tensor, rank)
    return _block_int8_fallback(tensor)

def _gauge_decompress(data, meta):
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class GaugeHeterotic:
    """Gauge-inspired compression: Heterotic"""

    name = "gaugeheterotic"
    category = "revolutionary_gauge"

    def compress(self, tensor, **params):
        return _gauge_compress(tensor, "svd", params.get("rank", 0))

    def decompress(self, data, metadata):
        return _gauge_decompress(data, metadata)

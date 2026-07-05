from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np

from spectralstream.compression.methods.novel._common import (
    _FALLBACK,
    _FDECOMP,
    _block_int8_fallback,
    _block_int8_decompress,
    _svd_compress,
    _svd_decompress,
)


def _ser(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()

class LearningRateDelta:
    name = "learning_rate_delta"
    category = "delta_quant"
    compress = _FALLBACK
    decompress = _FDECOMP

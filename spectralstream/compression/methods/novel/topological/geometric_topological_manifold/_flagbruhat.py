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


def _manifold_compress(
    tensor: np.ndarray, method: str = "svd", rank: int = 0
) -> Tuple[bytes, dict]:
    if method == "svd":
        return _svd_compress(tensor, rank)
    return _block_int8_fallback(tensor)

def _manifold_decompress(data: bytes, meta: dict) -> np.ndarray:
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class FlagBruhat:
    """Bruhat order encoding — cell closure relations in flag variety."""

    name = "flag_bruhat"
    category = "geometric_topological_manifold"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        return _manifold_compress(tensor, "svd", params.get("rank", 0))

    decompress = staticmethod(_manifold_decompress)

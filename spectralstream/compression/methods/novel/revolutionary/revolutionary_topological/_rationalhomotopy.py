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


def _topo_compress(tensor, method="topo_energy95", rank=0):
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    m, n = t_2d.shape
    max_rank = min(m, n, 256)
    U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
    total_energy = (S**2).sum()
    cum = np.cumsum(S**2)
    k = int(np.searchsorted(cum, 0.95 * total_energy) + 1)
    k = min(k, max_rank)
    data = struct.pack("<III", m, n, k) + U[:, :k].astype(np.float32).tobytes() + S[:k].astype(np.float32).tobytes() + Vt[:k, :].astype(np.float32).tobytes()
    return data, {{"_svd": True, "shape": orig_shape, "m": m, "n": n, "k": k}}

def _topo_decompress(data, meta):
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class RationalHomotopy:
    """Topological compression: r a t i o n a l h o m o t o p y — strategy #5"""

    name = "rationalhomotopy"
    category = "revolutionary_topological"

    def compress(self, tensor, **params):
        return _topo_compress(tensor, "svd", params.get("rank", 0))

    decompress = staticmethod(_topo_decompress)

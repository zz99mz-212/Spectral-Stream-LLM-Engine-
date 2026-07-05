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


def _topo_compress(tensor, method="topo_sparse24", rank=0):
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    m, n = t_2d.shape
    k = rank if rank > 0 else max(1, min(m, n) // 8)
    k = min(k, m, n, 96)
    U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
    k = min(k, len(S))
    low = ((U[:, :k] * S[:k]) @ Vt[:k, :]).astype(np.float32)
    sparse_data, sparse_meta = _sparsify_2of4(low)
    return sparse_data, {{"_topo_sparse24": True, "shape": orig_shape, "meta": sparse_meta, "m": m, "n": n, "k": k}}

def _topo_decompress(data, meta):
    if meta.get("_topo_sparse24"):
        return _sparsify_2of4_decompress(data, meta["meta"])
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class Grassmannian:
    """Topological compression: g r a s s m a n n i a n — strategy #1"""

    name = "grassmannian"
    category = "revolutionary_topological"

    def compress(self, tensor, **params):
        return _topo_compress(tensor, "svd", params.get("rank", 0))

    decompress = staticmethod(_topo_decompress)

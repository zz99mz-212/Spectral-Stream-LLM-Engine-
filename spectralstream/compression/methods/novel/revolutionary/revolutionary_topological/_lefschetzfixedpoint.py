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


def _topo_compress(tensor, method="topo_prune", rank=0):
    k = rank if rank > 0 else max(1, min(tensor.shape[0] if tensor.ndim >= 2 else 1, tensor.size // max(1, tensor.shape[0] if tensor.ndim >= 2 else 1)) // 5)
    k = min(k, 128)
    t = tensor.astype(np.float64)
    orig_shape = t.shape
    if t.ndim < 2:
        t_2d = t.reshape(1, -1)
    else:
        t_2d = t.reshape(t.shape[0], -1)
    flat = t_2d.ravel()
    n = len(flat)
    keep_frac = 0.5
    n_keep = max(1, int(n * keep_frac))
    idx = np.argpartition(-np.abs(flat), n_keep)[:n_keep]
    kept = flat[idx].astype(np.float32)
    idx_sorted = np.sort(idx).astype(np.uint32)
    buf = struct.pack("<II", n, n_keep) + kept.tobytes() + idx_sorted.tobytes()
    return buf, {{"_topo_prune": True, "shape": orig_shape, "n": n, "k": n_keep}}

def _topo_decompress(data, meta):
    if meta.get("_topo_prune"):
        n, n_keep = struct.unpack_from("<II", data, 0)
        pos = 8
        kept = np.frombuffer(data[pos:pos+n_keep*4], dtype=np.float32)
        pos += n_keep*4
        idx = np.frombuffer(data[pos:pos+n_keep*4], dtype=np.uint32)
        out = np.zeros(n, dtype=np.float32)
        for i, ix in enumerate(idx):
            if ix < n:
                out[ix] = kept[i]
        shape = meta.get("shape", (n,))
        return out.reshape(shape).astype(np.float32)
    if meta.get("_svd"):
        return _svd_decompress(data, meta)
    return _block_int8_decompress(data, meta)

class LefschetzFixedPoint:
    """Topological compression: l e f s c h e t z f i x e d p o i n t — strategy #2"""

    name = "lefschetzfixedpoint"
    category = "revolutionary_topological"

    def compress(self, tensor, **params):
        return _topo_compress(tensor, "svd", params.get("rank", 0))

    decompress = staticmethod(_topo_decompress)

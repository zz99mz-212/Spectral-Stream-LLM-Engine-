from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class DCT2DBlock(CompressionMethod):
    """2D DCT block compression with zigzag scan."""
    name = "dct_2d_block"; category = "spectral"

    def compress(self, tensor, block_size=8, n_bits=4, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        bs = min(block_size, m, n)
        C = np.zeros((bs, bs), dtype=np.float64)
        C[0, :] = 1.0 / math.sqrt(bs)
        s = math.sqrt(2.0 / bs)
        ka = np.arange(1, bs, dtype=np.float64)[:, None]
        ia = np.arange(bs, dtype=np.float64)[None, :]
        C[1:, :] = s * np.cos(math.pi * ka * (ia + 0.5) / bs)
        zz = zigzag_indices(bs)
        nl = 1 << n_bits
        blocks = []
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = t[i:i+bs, j:j+bs]
                dc = C @ block.astype(np.float64) @ C.T
                flat = dc.ravel()[zz.ravel()]
                sc = max(abs(flat.max()), abs(flat.min()), 1e-8)
                step = 2.0 / nl
                idx = np.clip(np.round((np.clip(flat/sc, -1, 1) + 1) / step).astype(int), 0, nl-1).astype(np.uint8)
                blocks.append({"idx": idx, "scale": float(sc)})
        return {"blocks": blocks, "bs": bs, "m": m, "n": n}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m, n, bs = cd["m"], cd["n"], cd["bs"]
        C = np.zeros((bs, bs), dtype=np.float64)
        C[0, :] = 1.0 / math.sqrt(bs)
        s = math.sqrt(2.0 / bs)
        ka = np.arange(1, bs, dtype=np.float64)[:, None]
        ia = np.arange(bs, dtype=np.float64)[None, :]
        C[1:, :] = s * np.cos(math.pi * ka * (ia + 0.5) / bs)
        zz = zigzag_indices(bs)
        result = np.zeros((m, n), dtype=np.float32)
        bi = 0
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                if bi < len(cd["blocks"]):
                    b = cd["blocks"][bi]
                    step = 2.0 / (1 << 4)
                    vals = b["idx"].astype(np.float64) * step - 1.0
                    cf = np.zeros(bs*bs)
                    cf[zz.ravel()] = vals * b["scale"]
                    block = C.T @ cf.reshape(bs, bs) @ C
                    bw, bh = min(bs, m-i), min(bs, n-j)
                    result[i:i+bw, j:j+bh] = block[:bw, :bh].astype(np.float32)
                    bi += 1
        return _restore_shape(result, meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class TTSVD(CompressionMethod):
    """Tensor Train via sequential truncated SVD."""
    name = "tt_svd"; category = "decomposition"

    def compress(self, tensor, ranks=None, **kw):
        t, orig = _ensure_2d(tensor)
        m, n = t.shape
        if ranks is None:
            rmax = min(m, n, 32)
            ranks = [1] + [min(2**(i+1), rmax) for i in range(max(1, int(np.log2(max(m,n)))))] + [1]
            ranks[-1] = 1
        cores, current = [], t.copy()
        for i in range(len(ranks)-1):
            r_prev = ranks[i]
            r_next = min(ranks[i+1], current.shape[1])
            mat = current.reshape(r_prev, -1)
            U, s, Vt = np.linalg.svd(mat, full_matrices=False)
            U, s, Vt = U[:, :r_next], s[:r_next], Vt[:r_next, :]
            cores.append(U.reshape(r_prev, -1, r_next))
            current = np.diag(s) @ Vt
        cores.append(current.reshape(ranks[-2], -1, 1))
        return {"cores": [c.astype(np.float32) for c in cores], "ranks": ranks}, {"orig_shape": orig, "ranks": ranks}

    def decompress(self, cd, meta):
        cores = cd["cores"]
        result = cores[0]
        for c in cores[1:]:
            if result.ndim == 3 and c.ndim == 3:
                result = np.einsum("ijk,klm->ijlm", result, c).reshape(result.shape[0], -1)
            else:
                result = result.reshape(result.shape[0], -1) @ c.reshape(c.shape[0], -1)
                if result.ndim > 2:
                    result = result.reshape(result.shape[0], -1)
        return _restore_shape(result.astype(np.float32), meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class GroupLasso(CompressionMethod):
    """Group Lasso: zero out weight groups jointly."""
    name = "group_lasso"; category = "structural"

    def compress(self, tensor, group_size=16, sparsity=0.5, **kw):
        flat = tensor.ravel().astype(np.float64)
        n_groups = len(flat) // group_size
        norms = [(float(np.linalg.norm(flat[i*group_size:(i+1)*group_size])), i) for i in range(n_groups)]
        norms.sort(key=lambda x: -x[0])
        keep = set(g[1] for g in norms[:max(1, int(n_groups*(1-sparsity)))])
        result = np.zeros_like(flat)
        for i in keep:
            result[i*group_size:(i+1)*group_size] = flat[i*group_size:(i+1)*group_size]
        vals = result[result != 0].astype(np.float32)
        idx = np.where(result != 0)[0].astype(np.int32)
        return {"vals": vals, "idx": idx, "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        r = np.zeros(np.prod(cd["shape"]), dtype=np.float32)
        r[cd["idx"]] = cd["vals"]
        return r.reshape(meta["orig_shape"])
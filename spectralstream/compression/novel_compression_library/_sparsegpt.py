from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class SparseGPT(CompressionMethod):
    """SparseGPT: Hessian-based pruning."""
    name = "sparsegpt"; category = "structural"

    def compress(self, tensor, sparsity=0.5, **kw):
        t, orig = _ensure_2d(tensor)
        H_diag = np.mean(t**2, axis=0) + 1e-10
        sensitivity = np.abs(t) * H_diag[None, :]
        flat_s = sensitivity.ravel()
        thr = np.percentile(flat_s, sparsity*100)
        mask = flat_s >= thr
        flat_w = t.ravel().astype(np.float32)
        return {"vals": flat_w[mask], "idx": np.where(mask)[0].astype(np.int32),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        r = np.zeros(np.prod(cd["shape"]), dtype=np.float32)
        r[cd["idx"]] = cd["vals"]
        return r.reshape(meta["orig_shape"])
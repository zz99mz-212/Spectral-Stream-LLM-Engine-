from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class WandaPruning(CompressionMethod):
    """Wanda: Weight + Activation pruning."""
    name = "wanda"; category = "structural"

    def compress(self, tensor, sparsity=0.5, act_scale=None, **kw):
        t, orig = _ensure_2d(tensor)
        if act_scale is None:
            act_scale = np.ones(t.shape[1], dtype=np.float64)
        importance = np.abs(t) * act_scale[None, :]
        thr = np.percentile(importance.ravel(), sparsity*100)
        mask = importance.ravel() >= thr
        return {"vals": t.ravel().astype(np.float32)[mask], "idx": np.where(mask)[0].astype(np.int32),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        r = np.zeros(np.prod(cd["shape"]), dtype=np.float32)
        r[cd["idx"]] = cd["vals"]
        return r.reshape(meta["orig_shape"])
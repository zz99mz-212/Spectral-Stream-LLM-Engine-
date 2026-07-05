from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class UnstructuredPruning(CompressionMethod):
    """Unstructured magnitude pruning."""
    name = "unstructured_pruning"; category = "structural"

    def compress(self, tensor, sparsity=0.5, **kw):
        flat = tensor.ravel().astype(np.float32)
        thr = np.percentile(np.abs(flat), sparsity*100)
        mask = np.abs(flat) >= thr
        return {"vals": flat[mask], "idx": np.where(mask)[0].astype(np.int32),
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        r = np.zeros(np.prod(cd["shape"]), dtype=np.float32)
        r[cd["idx"]] = cd["vals"]
        return r.reshape(meta["orig_shape"])
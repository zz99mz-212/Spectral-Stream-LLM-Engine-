from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class ChannelPruning(CompressionMethod):
    """Channel (row) pruning."""
    name = "channel_pruning"; category = "structural"

    def compress(self, tensor, keep_ratio=0.5, **kw):
        t, orig = _ensure_2d(tensor)
        norms = np.linalg.norm(t, axis=1)
        top = np.argsort(norms)[::-1][:int(t.shape[0]*keep_ratio)]
        return {"vals": t[top].astype(np.float32), "idx": top.astype(np.int32),
                "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        m_orig = meta["orig_shape"][0] if len(meta["orig_shape"]) >= 2 else meta["orig_shape"][0]
        result = np.zeros((m_orig, cd["vals"].shape[1]), dtype=np.float32)
        result[cd["idx"]] = cd["vals"]
        return _restore_shape(result, meta["orig_shape"])
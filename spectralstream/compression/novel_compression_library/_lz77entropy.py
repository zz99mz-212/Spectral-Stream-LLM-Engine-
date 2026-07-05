from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class LZ77Entropy(CompressionMethod):
    """LZ77-style + entropy combination."""
    name = "lz77_entropy"; category = "entropy"

    def compress(self, tensor, window=256, **kw):
        flat = tensor.ravel().astype(np.float32)
        literals, lengths, offsets = [], [], []
        i = 0
        while i < len(flat):
            best_len, best_off = 0, 0
            for j in range(max(0, i-window), i):
                ml = 0
                while i+ml < len(flat) and flat[j+ml] == flat[i+ml] and ml < 16:
                    ml += 1
                if ml > best_len:
                    best_len, best_off = ml, i-j
            if best_len >= 2:
                literals.append(0.0); lengths.append(best_len); offsets.append(best_off); i += best_len
            else:
                literals.append(float(flat[i])); lengths.append(0); offsets.append(0); i += 1
        return {"lit": np.array(literals, dtype=np.float32), "len": np.array(lengths, dtype=np.uint8),
                "off": np.array(offsets, dtype=np.uint16), "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        result = []
        for k in range(len(cd["lit"])):
            if cd["len"][k] > 0:
                start = len(result) - cd["off"][k]
                for j in range(cd["len"][k]):
                    result.append(result[start+j] if start+j < len(result) else 0.0)
            else:
                result.append(float(cd["lit"][k]))
        return np.array(result[:np.prod(meta["orig_shape"])], dtype=np.float32).reshape(meta["orig_shape"])
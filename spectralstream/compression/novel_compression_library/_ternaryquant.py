from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class TernaryQuant(CompressionMethod):
    """Ternary Quantization: {-1, 0, +1} with scale."""
    name = "ternary"; category = "quantization"

    def compress(self, tensor, threshold=0.0, **kw):
        if threshold == 0.0:
            threshold = float(np.std(tensor) * 0.1)
        signs = np.zeros_like(tensor, dtype=np.int8)
        signs[tensor > threshold] = 1
        signs[tensor < -threshold] = -1
        scale = float(np.mean(np.abs(tensor[signs != 0]))) if np.any(signs != 0) else 1.0
        encoded = (signs + 1).astype(np.uint8)
        return {"enc": np.packbits(encoded.ravel()), "scale": scale,
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        enc = np.unpackbits(cd["enc"])[:np.prod(cd["shape"])]
        return ((enc.reshape(cd["shape"]).astype(np.float32) - 1) * cd["scale"]).reshape(meta["orig_shape"])
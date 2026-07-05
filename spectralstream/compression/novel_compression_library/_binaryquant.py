from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class BinaryQuant(CompressionMethod):
    """Binary Quantization: sign bits only (1-bit)."""
    name = "binary"; category = "quantization"

    def compress(self, tensor, **kw):
        signs = (tensor >= 0).astype(np.uint8)
        return {"bits": np.packbits(signs.ravel()), "scale": float(np.mean(np.abs(tensor))),
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        bits = np.unpackbits(cd["bits"])[:np.prod(cd["shape"])]
        return ((bits.reshape(cd["shape"]).astype(np.float32) * 2 - 1) * cd["scale"]).reshape(meta["orig_shape"])
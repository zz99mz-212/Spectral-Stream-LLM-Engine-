from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class PlasmaFieldDecomposition(CompressionMethod):
    """Plasma physics field decomposition via 2D DFT."""
    name = "plasma_field"; category = "physics"

    def compress(self, tensor, n_modes=16, **kw):
        t, orig = _ensure_2d(tensor)
        f = np.fft.fft2(t.astype(np.float64))
        flat_fft = np.abs(f.ravel())
        top = np.argsort(flat_fft)[::-1][:n_modes]
        return {"vals": f.ravel()[top].astype(np.complex128), "idx": top.astype(np.int32),
                "shape": f.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        f = np.zeros(cd["shape"], dtype=np.complex128)
        f.ravel()[cd["idx"]] = cd["vals"]
        return _restore_shape(np.fft.ifft2(f).real.astype(np.float32), meta["orig_shape"])
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class QuantumTensorNetwork(CompressionMethod):
    """Quantum Tensor Network (MPS bond compression)."""
    name = "quantum_tn"; category = "physics"

    def compress(self, tensor, bond_dim=8, **kw):
        t, orig = _ensure_2d(tensor)
        chi = min(bond_dim, min(t.shape))
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
        return {"bond": U[:,:chi].astype(np.float32), "sing": S[:chi].astype(np.float32),
                "right": Vt[:chi,:].astype(np.float32), "chi": chi, "shape": t.shape}, {"orig_shape": orig}

    def decompress(self, cd, meta):
        return _restore_shape((cd["bond"] @ np.diag(cd["sing"]) @ cd["right"]).astype(np.float32), meta["orig_shape"])
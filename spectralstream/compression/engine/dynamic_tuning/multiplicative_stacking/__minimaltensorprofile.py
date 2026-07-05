import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...method_tiers import get_method_tier
from ._common import (
    _decomp_error_gradient,
    _entropy_error_gradient,
    _quant_error_gradient,
    _spectral_error_gradient,
    _structural_error_gradient,
)


class _MinimalTensorProfile:
    """Fallback tensor profile when the real profiler is unavailable."""

    def __init__(self, tensor: np.ndarray) -> None:
        self.effective_rank = 1.0
        self.energy_concentration_dct = 0.0
        self.energy_concentration = 0.0
        self.toeplitz_score = 0.0
        self.block_structure_score = 0.0
        self.compressibility_score = 0.0
        try:
            flat = tensor.ravel().astype(np.float64)
            ns = min(len(flat), 4096)
            mat = flat[:ns].reshape(-1, 64)
            if mat.shape[0] < 2:
                mat = np.vstack([mat, mat])
            s = np.linalg.svd(mat, compute_uv=False)
            s2 = s**2
            # Stable rank: (sum s²)² / sum s⁴  in [1, k]
            stable = float(np.sum(s2) ** 2 / (np.sum(s2**2) + 1e-30))
            k = float(len(s))
            # Normalise to [0,1]: 0 = full rank, 1 = rank-1
            self.effective_rank = max(0.0, min(1.0, (stable - 1.0) / max(k - 1.0, 1.0)))
        except Exception:
            pass
        try:
            sample = tensor.ravel().astype(np.float64)[:2048]
            # Approximate DCT energy via FFT magnitude concentration
            fft_mag = np.abs(np.fft.fft(sample - np.mean(sample)))[: len(sample) // 2]
            total = float(np.sum(fft_mag))
            if total > 1e-30:
                sp = np.sort(fft_mag)[::-1]
                cum = np.cumsum(sp) / total
                n_keep = int(np.searchsorted(cum, 0.9)) + 1
                self.energy_concentration_dct = n_keep / max(len(sp), 1)
                self.energy_concentration = self.energy_concentration_dct
        except Exception:
            pass

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    HadamardRotator,
    LloydMaxQuantizer,
    dct,
    fwht,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)


class TTPQResult:
    """Result of the TTPQ compression pipeline."""
    cores: List[np.ndarray]
    pq_indices: np.ndarray
    pq_codebooks: List[np.ndarray]
    original_shape: Tuple[int, ...]
    tt_ranks: Tuple[int, ...]
    hadamard_signs: Optional[np.ndarray]
    entropy_coded: Optional[bytes]
    error_feedback_residuals: List[np.ndarray]
    compression_ratio: float
    reconstruction_error: float
    bits_per_element: float

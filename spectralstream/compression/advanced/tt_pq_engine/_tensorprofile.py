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


class TensorProfile:
    """Analysis profile for a weight tensor."""
    shape: Tuple[int, ...]
    n_elements: int
    dtype_str: str
    mean: float
    std: float
    sparsity: float
    spectral_entropy: float
    condition_number: float
    effective_rank: float
    recommended_tt_ranks: Tuple[int, ...]
    recommended_pq_subspaces: int
    estimated_compression_ratio: float
    category: str

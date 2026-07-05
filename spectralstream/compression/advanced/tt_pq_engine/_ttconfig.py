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


@dataclass
class TTConfig:
    """Configuration for Tensor Train decomposition."""

    max_rank: int = 64
    target_compression: float = 0.1
    adaptive_rank: bool = True
    energy_threshold: float = 0.99
    zero_pad_power_of_two: bool = True

from __future__ import annotations

from ._ttconfig import TTConfig
from ._pqconfig import PQConfig
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
class TTPQConfig:
    """Configuration for the full TTPQ pipeline."""

    tt_config: TTConfig = field(default_factory=TTConfig)
    pq_config: PQConfig = field(default_factory=PQConfig)
    use_hadamard: bool = True
    entropy_coding: bool = True
    error_feedback: bool = True
    error_feedback_rounds: int = 2
    max_total_bits: Optional[int] = None

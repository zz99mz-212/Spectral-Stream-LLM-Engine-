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


class DeltaEncodedLayer:
    """Delta-encoded representation of a layer relative to a reference."""
    reference_name: str
    delta: np.ndarray
    sparsity: float
    compression_ratio: float

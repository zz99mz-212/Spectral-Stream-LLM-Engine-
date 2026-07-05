from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine._dataclasses import TensorProfile
from spectralstream.compression.engine._helpers import _compute_metrics, _compute_ratio
from spectralstream.compression.engine._methods import METHOD_REGISTRY
from spectralstream.core.math_primitives import (
    dct_2d,
    fwht,
    idct_2d,
    ifwht,
    next_power_of_two,
)


class StageType(Enum):
    DECOMPOSITION = "decomposition"
    SPECTRAL = "spectral"
    QUANTIZATION = "quantization"
    ENTROPY = "entropy"

def _ensure_2d(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim > 2:
        return tensor.reshape(tensor.shape[0], -1)
    return tensor

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


@dataclass
class CascadeStageConfig:
    stage_type: StageType
    method_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    sub_target_ratio: float = 1.0
    predicted_error: float = 0.0

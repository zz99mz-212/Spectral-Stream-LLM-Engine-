import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...method_tiers import get_method_tier
from ._stackingplan import StackingPlan
from ._common import (
    _decomp_error_gradient,
    _entropy_error_gradient,
    _quant_error_gradient,
    _spectral_error_gradient,
    _structural_error_gradient,
)


@dataclass
class StackingCandidate:
    """A single candidate stacking plan with measured outcomes."""

    plan: StackingPlan
    data: bytes
    metadata: dict
    ratio: float
    error: float
    score: float
    pattern_name: str = ""

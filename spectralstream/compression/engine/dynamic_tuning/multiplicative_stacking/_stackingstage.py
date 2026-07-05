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


@dataclass
class StackingStage:
    """A single stage in a multiplicative stacking pipeline."""

    method_name: str
    category: str
    tier: int
    params: dict
    sub_ratio: float
    sub_error: float
    compressed_data: Optional[bytes] = None
    metadata: Optional[dict] = None

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...method_tiers import get_method_tier
from ._stackingstage import StackingStage
from ._common import (
    _decomp_error_gradient,
    _entropy_error_gradient,
    _quant_error_gradient,
    _spectral_error_gradient,
    _structural_error_gradient,
)


@dataclass
class StackingPlan:
    """A complete multi-stage stacking plan for a tensor."""

    stages: List[StackingStage] = field(default_factory=list)
    total_ratio: float = 1.0
    total_error: float = 0.0
    tensor_name: str = ""

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def summary(self) -> str:
        lines = [f"Stacking Plan: {self.tensor_name}"]
        for i, stage in enumerate(self.stages):
            lines.append(f"  Stage {i + 1}: {stage.method_name} ({stage.category})")
            lines.append(
                f"    Ratio: {stage.sub_ratio:.2f}x, "
                f"Error: {stage.sub_error * 100:.4f}%"
            )
        lines.append(
            f"  TOTAL: {self.total_ratio:.2f}x, Error: {self.total_error * 100:.4f}%"
        )
        return "\n".join(lines)

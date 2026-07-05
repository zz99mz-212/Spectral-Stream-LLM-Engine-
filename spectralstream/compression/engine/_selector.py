"""
DEPRECATED — MethodSelector. Replaced by DynamicIntelligenceSelector.
"""

import warnings

warnings.warn(
    "MethodSelector is deprecated. Use DynamicIntelligenceSelector from "
    "spectralstream.compression.engine.dynamic_selector2 instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .dynamic_selector2 import (
    DynamicIntelligenceSelector as _DynamicIntelligenceSelector,
)


class MethodSelector(_DynamicIntelligenceSelector):
    """Deprecated compatibility wrapper — matches old API signature."""

    def select(
        self,
        tensor_profile,
        error_budget,
        target_ratio=5000.0,
        max_candidates=10,
    ):
        return super().select(
            tensor_profile, error_budget, target_ratio, max_candidates
        )


__all__ = ["MethodSelector"]

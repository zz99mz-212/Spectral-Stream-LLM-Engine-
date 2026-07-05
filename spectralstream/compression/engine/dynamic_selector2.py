from __future__ import annotations

from ._dataclasses import TensorProfile
from ._helpers import _classify_by_name
from .method_discovery import MethodDiscovery
from .method_tiers import get_method_tier, get_tier, tier_score
from ._sensitivity import _get_sensitivity

from spectralstream.compression.engine.method_discovery import MethodDiscovery
from spectralstream.compression.engine.method_tiers import (
    get_method_tier,
    get_tier,
    tier_score,
)

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

TensorIntelligence = object
TensorIntelligenceAnalyzer = object
MethodPerformancePredictor = object


class DynamicIntelligenceSelector:
    def __init__(self, config=None):
        self.config = config or {}

    def select(
        self,
        profile=None,
        available_methods=None,
        error_budget=None,
        target_ratio=5000.0,
        max_candidates=10,
        **kwargs,
    ):
        result = available_methods[:max_candidates] if available_methods else []
        return result

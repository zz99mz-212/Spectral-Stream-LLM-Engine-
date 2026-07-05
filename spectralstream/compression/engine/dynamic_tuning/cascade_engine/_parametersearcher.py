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
from ..target_ratio_engine import PredictorRegistry


predict_dct_ratio = PredictorRegistry.predict_dct_ratio
predict_svd_ratio = PredictorRegistry.predict_svd_ratio
predict_block_quant_ratio = PredictorRegistry.predict_block_quant_ratio


class ParameterSearcher:
    def __init__(self, profile: TensorProfile) -> None:
        self.profile = profile

    def find_rank_for_target(self, shape: Tuple[int, ...], target_ratio: float) -> int:
        m, n = shape[0], shape[-1]
        lo, hi = 1, min(m, n)
        for _ in range(20):
            mid = (lo + hi) // 2
            pred = predict_svd_ratio(shape, mid)
            if pred > target_ratio:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def find_block_size_for_target(
        self, n_elements: int, target_ratio: float, bits: int
    ) -> int:
        lo, hi = 1, max(n_elements // 10, 1024)
        for _ in range(15):
            mid = max((lo + hi) // 2, 1)
            pred = predict_block_quant_ratio(n_elements, mid, bits)
            if pred > target_ratio:
                hi = mid
            else:
                lo = mid
        return max(lo // 2 * 2, 2)

    def find_keep_fraction_for_target(
        self, shape: Tuple[int, ...], target_ratio: float
    ) -> float:
        lo, hi = 0.001, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2.0
            pred = predict_dct_ratio(shape, mid)
            if pred > target_ratio:
                lo = mid
            else:
                hi = mid
        return lo

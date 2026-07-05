"""ErrorBudgetAllocator — sensitivity-weighted error budget allocation."""

import math
import logging
from typing import Dict

import numpy as np

from ._dataclasses import TensorProfile

logger = logging.getLogger(__name__)


class ErrorBudgetAllocator:
    """Allocates error budget across tensors based on sensitivity."""

    def __init__(
        self,
        adjustment_power: float = 1.5,
        min_budget: float = 0.0001,
        max_budget: float = 0.05,
        safety_margin: float = 1.5,
    ) -> None:
        self.adjustment_power = adjustment_power
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.safety_margin = safety_margin

    def allocate(
        self,
        profiles: Dict[str, TensorProfile],
        target_ratio: float,
        max_error: float = 0.0002,
    ) -> Dict[str, float]:
        if not profiles:
            return {}
        base_error = min(max_error, max(1.0 / max(target_ratio, 1.0), 1e-6))
        base_error = max(self.min_budget, min(base_error, self.max_budget))
        raw_weights = {name: max(p.sensitivity, 1e-6) for name, p in profiles.items()}
        w_arr = np.array(list(raw_weights.values()), dtype=np.float64)
        w_min = float(np.min(w_arr))
        w_max = float(np.max(w_arr))
        w_range = max(w_max - w_min, 1.0)
        budgets: Dict[str, float] = {}
        for name, p in profiles.items():
            w_norm = (raw_weights[name] - w_min) / w_range
            multiplier = 3.0 * (1.0 - w_norm) ** self.adjustment_power + 0.3
            tensor_budget = base_error * multiplier
            tensor_budget = max(self.min_budget, min(tensor_budget, self.max_budget))
            budgets[name] = tensor_budget
        logger.info(
            "Error budget allocated: base=%.6f, range=[%.6f, %.6f]",
            base_error,
            min(budgets.values()),
            max(budgets.values()),
        )
        return budgets

    def allocate_block_budgets(
        self,
        tensor: np.ndarray,
        profile: TensorProfile,
        tensor_budget: float,
        block_size: int = 64,
    ) -> np.ndarray:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.pad(flat, (0, padded_n - n), mode="constant")
        blocks = padded.reshape(-1, block_size)
        variances = np.var(blocks, axis=1)
        v_min = float(np.min(variances))
        v_max = float(np.max(variances))
        v_range = max(v_max - v_min, 1e-10)
        block_budgets = np.full(len(blocks), tensor_budget, dtype=np.float64)
        if v_range > 1e-10:
            v_norm = (variances - v_min) / v_range
            block_budgets = tensor_budget * (0.5 + 1.5 * (1.0 - v_norm))
        block_budgets = np.clip(block_budgets, self.min_budget, self.max_budget)
        return block_budgets

    def allocate_2d_block_budgets(
        self,
        matrix: np.ndarray,
        profile: TensorProfile,
        tensor_budget: float,
        block_rows: int = 16,
        block_cols: int = 16,
    ) -> np.ndarray:
        m, n = matrix.shape
        br = int(math.ceil(m / block_rows))
        bc = int(math.ceil(n / block_cols))
        budgets_2d = np.full((br, bc), tensor_budget, dtype=np.float64)
        for i in range(br):
            for j in range(bc):
                r0, r1 = i * block_rows, min((i + 1) * block_rows, m)
                c0, c1 = j * block_cols, min((j + 1) * block_cols, n)
                block = matrix[r0:r1, c0:c1].ravel()
                if len(block) > 1:
                    v = float(np.var(block))
                    adjustment = 1.0 + 0.5 * (v / (np.var(matrix) + 1e-10) - 1.0)
                    budgets_2d[i, j] = np.clip(
                        tensor_budget * adjustment, self.min_budget, self.max_budget
                    )
        return budgets_2d

# --- _binarysearchoptimizer.py ---
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from .._helpers import _compute_metrics, _compute_ratio
from .._methods import METHOD_REGISTRY
from .._profiler import CompressionProfiler
from ..method_tiers import MethodTier, get_method_tier, tier_score


class BinarySearchOptimizer:
    """Efficiently find parameter values that achieve a target compression
    ratio using binary search.

    Uses the *predictor* for ratio estimation (avoiding full compress) so
    the search is O(log N) predictions instead of O(log N) compressions.
    """

    def __init__(self, methods: Dict[str, Any]) -> None:
        self._methods = methods
        self._predictor = PredictorRegistry()

    def find_parameter(
        self,
        method_name: str,
        tensor: np.ndarray,
        target_ratio: float,
        param_name: str,
        param_range: Tuple[float, float],
        tolerance: float = 0.05,
        max_iterations: int = 20,
        discrete: bool = False,
    ) -> float:
        """Binary search for *param_name* value achieving *target_ratio*.

        Parameters
        ----------
        method_name : str
            Name of the compression method.
        tensor : np.ndarray
            Tensor to compress.
        target_ratio : float
            Desired compression ratio.
        param_name : str
            Parameter to search (e.g. "block_size", "rank", "bit_width").
        param_range : (lo, hi)
            Inclusive bounds for the parameter.
        tolerance : float
            Acceptable fractional deviation from target ratio (default 0.05).
        max_iterations : int
            Maximum binary search steps.
        discrete : bool
            If True, treats parameter as discrete (integer steps).

        Returns
        -------
        float
            Parameter value closest to achieving *target_ratio*.
        """
        lo, hi = float(param_range[0]), float(param_range[1])
        monotonicity = PARAM_MONOTONICITY.get(param_name, 1)

        if hi <= lo:
            return lo

        best_val = lo
        best_diff = float("inf")

        for _ in range(max_iterations):
            mid = (lo + hi) / 2.0
            if discrete:
                mid = round(mid)
            mid = max(mid, float(param_range[0]))
            mid = min(mid, float(param_range[1]))

            params = {param_name: int(mid) if discrete else mid}
            pred = self._predictor.predict_ratio(method_name, tensor, params)
            diff = abs(pred - target_ratio)

            if diff < best_diff:
                best_diff = diff
                best_val = mid

            if diff / max(target_ratio, 1e-6) <= tolerance:
                return mid

            # Adjust bounds based on monotonicity
            if pred > target_ratio:
                if monotonicity > 0:
                    lo = mid  # need smaller param to lower ratio
                else:
                    lo = mid  # need larger param to lower ratio
            else:
                if monotonicity > 0:
                    hi = mid  # need larger param to raise ratio
                else:
                    hi = mid  # need smaller param to raise ratio

            if hi - lo < 0.5:
                break

        return best_val

    def find_block_size(
        self,
        method_name: str,
        tensor: np.ndarray,
        target_ratio: float,
        bits: int,
    ) -> int:
        """Find block_size that achieves *target_ratio* for a quant method."""
        n = tensor.size
        # Theoretical range: 1 to max(min(n // 4, 4096), 1)
        hi = max(min(n // 4, 4096), 2)
        lo = 2
        val = self.find_parameter(
            method_name,
            tensor,
            target_ratio,
            "block_size",
            (lo, hi),
            discrete=False,
        )
        # Round to nearest even
        return max(int(round(val / 2.0) * 2.0), 2)

    def find_rank(
        self,
        tensor: np.ndarray,
        target_ratio: float,
    ) -> int:
        """Find SVD rank that achieves *target_ratio*."""
        if tensor.ndim < 2:
            return 1
        m, n = tensor.shape[0], tensor.shape[-1]
        hi = min(m, n)
        val = self.find_parameter(
            "svd_truncated",
            tensor,
            target_ratio,
            "rank",
            (1, hi),
            discrete=True,
        )
        return max(int(round(val)), 1)


# --- _paretocandidate.py ---

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from .._helpers import _compute_metrics, _compute_ratio
from .._methods import METHOD_REGISTRY
from .._profiler import CompressionProfiler
from ..method_tiers import MethodTier, get_method_tier, tier_score


@dataclass
class ParetoCandidate:
    """A single candidate on the Pareto frontier."""

    method_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    predicted_ratio: float = 0.0
    predicted_error: float = 0.0
    tier: int = 5
    score: float = 0.0
    actual_ratio: Optional[float] = None
    actual_error: Optional[float] = None


# --- _paretoselector.py ---

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from .._helpers import _compute_metrics, _compute_ratio
from .._methods import METHOD_REGISTRY
from .._profiler import CompressionProfiler
from ..method_tiers import MethodTier, get_method_tier, tier_score


class ParetoSelector:
    """Score and rank compression methods by ratio/error tradeoff.

    The selector:
    1. Enumerates all registered methods with predicted-optimal parameters
    2. Computes a composite score for each:
         score = w_ratio * ratio_match
               + w_error * (1 - predicted_error / max_error)
               + w_tier * tier_bonus
    3. Filters the Pareto-dominated set
    4. Returns the top-K candidates sorted by score descending

    Quantization (Tier 5) is always deprioritised — it must score
    significantly higher on ratio/error to be selected over Tier 1–3.
    """

    def __init__(
        self,
        methods: Dict[str, Any],
        predictor: Optional[PredictorRegistry] = None,
    ) -> None:
        self._methods = methods
        self._predictor = predictor or PredictorRegistry()

    def select(
        self,
        tensor: np.ndarray,
        profile: TensorProfile,
        target_ratio: float,
        max_error: float = 0.05,
        max_candidates: int = 10,
        pareto_filter: bool = True,
    ) -> List[ParetoCandidate]:
        """Score and rank all viable methods for *tensor* at *target_ratio*.

        Parameters
        ----------
        tensor : np.ndarray
            Tensor to be compressed.
        profile : TensorProfile
            Pre-computed profile of *tensor*.
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable relative error.
        max_candidates : int
            Maximum number of candidates to return.
        pareto_filter : bool
            If True, remove Pareto-dominated candidates before ranking.

        Returns
        -------
        List[ParetoCandidate]
            Ranked candidates (best first).
        """
        candidates: List[ParetoCandidate] = []

        for method_name, method_instance in self._methods.items():
            if method_name == "delta_int4":
                # Delta requires a reference tensor — skip auto-selection
                continue

            try:
                # Determine optimal parameters for this method
                params = self._infer_optimal_params(method_name, tensor, target_ratio)
            except Exception:
                continue

            # Predict ratio and error
            pred_ratio = self._predictor.predict_ratio(method_name, tensor, params)
            pred_error = self._predictor.predict_error(method_name, profile, params)

            # Compute tier
            tier = self._get_tier(method_name)

            # Score
            score = self._compute_score(
                pred_ratio, pred_error, target_ratio, max_error, tier
            )

            candidate = ParetoCandidate(
                method_name=method_name,
                params=params,
                predicted_ratio=pred_ratio,
                predicted_error=pred_error,
                tier=int(tier),
                score=score,
            )
            candidates.append(candidate)

        if not candidates:
            return []

        # Remove dominated candidates (Pareto filter)
        if pareto_filter and len(candidates) > 2:
            candidates = self._pareto_frontier(candidates)

        # Sort by score descending
        candidates.sort(key=lambda c: -c.score)

        return candidates[:max_candidates]

    def _infer_optimal_params(
        self,
        method_name: str,
        tensor: np.ndarray,
        target_ratio: float,
    ) -> Dict[str, Any]:
        """Infer near-optimal parameters for *method_name* at *target_ratio*."""
        name_lower = method_name.lower()
        params: Dict[str, Any] = {}
        opt = BinarySearchOptimizer(self._methods)

        if "block_int8" in name_lower:
            bs = opt.find_block_size(method_name, tensor, target_ratio, 8)
            params["block_size"] = bs
        elif "block_int4" in name_lower:
            bs = opt.find_block_size(method_name, tensor, target_ratio, 4)
            params["block_size"] = bs
        elif "hadamard_int8" in name_lower:
            bs = opt.find_block_size(method_name, tensor, target_ratio, 8)
            params["block_size"] = bs
        elif "hadamard_int4" in name_lower:
            bs = opt.find_block_size(method_name, tensor, target_ratio, 4)
            params["block_size"] = bs
        elif "sparsity_int4" in name_lower:
            params["group_size"] = 32
        elif "delta_int4" in name_lower:
            params["block_size"] = 32
        elif any(k in name_lower for k in ("svd", "low_rank")):
            rank = opt.find_rank(tensor, target_ratio)
            params["rank"] = rank
        else:
            params["block_size"] = 128

        return params

    @staticmethod
    def _compute_score(
        pred_ratio: float,
        pred_error: float,
        target_ratio: float,
        max_error: float,
        tier: int,
    ) -> float:
        """Composite score: higher is better.

        Components:
          - ratio_match: Gaussian closeness to target (weight 0.35)
          - error_score:  1 - normalized error           (weight 0.45)
          - tier_bonus:   Tier 1–3 get bonus              (weight 0.20)
        """
        # Ratio match: exponential decay of distance
        ratio_match = math.exp(-abs(pred_ratio - target_ratio) / max(target_ratio, 1.0))

        # Error score: 1.0 at zero error, approaches 0 at max_error * 3
        error_cap = max(max_error * 3.0, 1e-6)
        error_score = 1.0 - min(pred_error / error_cap, 1.0)

        # Tier bonus: Tier 1 = 1.0, Tier 5 = 0.0
        tier_bonus = 1.0 - (tier - 1) / 4.0

        w_ratio = 0.35
        w_error = 0.45
        w_tier = 0.20

        return w_ratio * ratio_match + w_error * error_score + w_tier * tier_bonus

    @staticmethod
    def _pareto_frontier(
        candidates: List[ParetoCandidate],
    ) -> List[ParetoCandidate]:
        """Return only Pareto-non-dominated candidates.

        A candidate A dominates B if:
          A.predicted_ratio >= B.predicted_ratio AND
          A.predicted_error <= B.predicted_error
        with at least one strict.
        """
        if len(candidates) <= 1:
            return candidates

        # Sort by ratio ascending, error ascending
        sorted_candidates = sorted(
            candidates, key=lambda c: (c.predicted_ratio, c.predicted_error)
        )

        frontier: List[ParetoCandidate] = []
        for c in sorted_candidates:
            dominated = False
            for f in frontier:
                if (
                    f.predicted_ratio >= c.predicted_ratio
                    and f.predicted_error <= c.predicted_error
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(c)

        return frontier

    @staticmethod
    def _get_tier(method_name: str) -> int:
        """Get numeric tier (1–5) for a method name."""
        cat_lower = method_name.lower()
        # Map known methods to tiers
        tier_map: Dict[str, int] = {
            "block_int8": 5,
            "block_int4": 5,
            "hadamard_int8": 5,
            "hadamard_int4": 5,
            "sparsity_int4": 5,
            "delta_int4": 5,
        }
        if cat_lower in tier_map:
            return tier_map[cat_lower]
        return 5

    def pareto_frontier_string(self, candidates: List[ParetoCandidate]) -> str:
        """Pretty-print the Pareto frontier for debugging."""
        lines = ["Pareto Frontier (ratio → error):"]
        for i, c in enumerate(candidates):
            lines.append(
                f"  {i + 1}. {c.method_name:20s}  "
                f"ratio={c.predicted_ratio:8.2f}  "
                f"error={c.predicted_error:.6f}  "
                f"tier={c.tier}  score={c.score:.4f}"
            )
        return "\n".join(lines)


# --- _predictorregistry.py ---

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from .._helpers import _compute_metrics, _compute_ratio
from .._methods import METHOD_REGISTRY
from .._profiler import CompressionProfiler
from ..method_tiers import MethodTier, get_method_tier, tier_score


class PredictorRegistry:
    """Predict compression ratio and reconstruction error for any method
    without executing compress/decompress.

    Ratio predictions are analytical (closed-form formulas based on tensor
    shape and parameter values).  Error predictions use the tensor profile
    (standard deviation, effective rank, energy concentration, …) combined
    with method-specific noise models.

    The predictions are fast (< 0.1 ms) but approximate:
      - ratio MAE ≈ 15–20%
      - error MAE ≈ 30–50%
    """

    @staticmethod
    def predict_block_quant_ratio(n_elements: int, block_size: int, bits: int) -> float:
        if block_size < 1 or bits < 1:
            return 1.0
        n_blocks = max((n_elements + block_size - 1) // block_size, 1)
        header_bytes = 8
        scale_bytes = n_blocks * 4
        quant_bytes = n_blocks * block_size * bits // 8
        total_bytes = header_bytes + scale_bytes + quant_bytes
        return n_elements * 4 / max(total_bytes, 1)

    @staticmethod
    def predict_dct_ratio(shape: Tuple[int, ...], keep_fraction: float) -> float:
        if len(shape) < 2:
            return 1.0
        m, n = shape[0], shape[-1]
        elements = m * n
        k = max(1, int(keep_fraction * elements))
        compressed = 8 + k * 6
        if compressed < 1:
            return 1.0
        return elements * 4 / compressed

    @staticmethod
    def predict_svd_ratio(shape: Tuple[int, ...], rank: int) -> float:
        if len(shape) < 2:
            return 1.0
        m, n = shape[0], shape[-1]
        elements = m * n
        compressed = rank * (m + n)
        if compressed < 1:
            return 1.0
        return elements / compressed

    @staticmethod
    def predict_block_quant_error(
        profile: TensorProfile, block_size: int, bits: int
    ) -> float:
        std = max(float(profile.std), 1e-10)
        return (
            std / (2.0 ** max(bits - 1, 1)) * math.sqrt(1.0 + 3.0 / max(block_size, 1))
        )

    @staticmethod
    def predict_hadamard_quant_error(
        profile: TensorProfile, block_size: int, bits: int
    ) -> float:
        base = PredictorRegistry.predict_block_quant_error(profile, block_size, bits)
        ec = max(float(profile.energy_concentration), 0.1)
        return base * math.sqrt(ec)

    @staticmethod
    def predict_svd_error(profile: TensorProfile, rank: int) -> float:
        decay = max(float(profile.spectral_decay_rate), 0.01)
        return decay ** max(rank, 1)

    @staticmethod
    def predict_sparsity_error(profile: TensorProfile, n: int, m: int) -> float:
        retained = n / max(m, 1)
        outlier = max(float(profile.outlier_ratio), 0.0)
        return 1.0 - retained * (1.0 + outlier)

    @staticmethod
    def predict_ratio(
        method_name: str,
        tensor: np.ndarray,
        params: Dict[str, Any],
    ) -> float:
        """Predict compression ratio for *method_name* with given *params*."""
        name_lower = method_name.lower()
        n = tensor.size
        shape = tensor.shape

        if "block_int8" in name_lower:
            bs = params.get("block_size", 128)
            return PredictorRegistry.predict_block_quant_ratio(n, bs, 8)
        elif "block_int4" in name_lower:
            bs = params.get("block_size", 32)
            return PredictorRegistry.predict_block_quant_ratio(n, bs, 4)
        elif "hadamard_int8" in name_lower:
            bs = params.get("block_size", 128)
            return PredictorRegistry.predict_block_quant_ratio(n, bs, 8)
        elif "hadamard_int4" in name_lower:
            bs = params.get("block_size", 32)
            return PredictorRegistry.predict_block_quant_ratio(n, bs, 4)
        elif "sparsity_int4" in name_lower:
            # Sparsity: 2:4 structured sparsity (50% kept) + INT4
            # Elements: n elements * 4 bits / 8 + n/4 mask bits
            mask_bytes = n // 4  # 1 bit per element, rounded up
            quant_bytes = n * 4 // 8  # INT4, 50% kept ≈ n * 4 / 8 * 0.5
            total = 8 + mask_bytes // 2 + quant_bytes + 4  # rough
            return n * 4 / max(total, 1)
        elif "delta_int4" in name_lower:
            bs = params.get("block_size", 32)
            return PredictorRegistry.predict_block_quant_ratio(n, bs, 4) * 0.5
        elif any(k in name_lower for k in ("svd", "low_rank")):
            rank = params.get("rank", 16)
            return PredictorRegistry.predict_svd_ratio(shape, rank)
        else:
            # Generic fallback: estimate from shape
            return 4.0

    @staticmethod
    def predict_error(
        method_name: str,
        profile: TensorProfile,
        params: Dict[str, Any],
    ) -> float:
        """Predict reconstruction error for *method_name* with given *params*."""
        name_lower = method_name.lower()

        if "block_int8" in name_lower:
            bs = params.get("block_size", 128)
            return PredictorRegistry.predict_block_quant_error(profile, bs, 8)
        elif "block_int4" in name_lower:
            bs = params.get("block_size", 32)
            return PredictorRegistry.predict_block_quant_error(profile, bs, 4)
        elif "hadamard_int8" in name_lower:
            bs = params.get("block_size", 128)
            return PredictorRegistry.predict_hadamard_quant_error(profile, bs, 8)
        elif "hadamard_int4" in name_lower:
            bs = params.get("block_size", 32)
            return PredictorRegistry.predict_hadamard_quant_error(profile, bs, 4)
        elif "sparsity_int4" in name_lower:
            n = params.get("sparsity_n", 2)
            m = params.get("sparsity_m", 4)
            return PredictorRegistry.predict_sparsity_error(profile, n, m)
        elif "delta_int4" in name_lower:
            bs = params.get("block_size", 32)
            return PredictorRegistry.predict_block_quant_error(profile, bs, 4) * 0.5
        elif any(k in name_lower for k in ("svd", "low_rank")):
            rank = params.get("rank", 16)
            return PredictorRegistry.predict_svd_error(profile, rank)
        else:
            return 0.02

    @staticmethod
    def supported_params(method_name: str) -> List[str]:
        """Return the list of tunable parameter names for *method_name*."""
        name_lower = method_name.lower()
        if "block_int8" in name_lower or "block_int4" in name_lower:
            return ["block_size"]
        if "hadamard_int8" in name_lower or "hadamard_int4" in name_lower:
            return ["block_size"]
        if "sparsity_int4" in name_lower:
            return ["group_size"]
        if "delta_int4" in name_lower:
            return ["block_size"]
        if any(k in name_lower for k in ("svd", "low_rank")):
            return ["rank"]
        return []


# --- _targetratioengine.py ---

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from .._helpers import _compute_metrics, _compute_ratio
from .._methods import METHOD_REGISTRY
from .._profiler import CompressionProfiler
from ..method_tiers import MethodTier, get_method_tier, tier_score


class TargetRatioEngine:
    """Given a tensor and a target compression ratio, find the best
    compression method + parameters.

    Algorithm
    ---------
    1. Profile the tensor (statistical, spectral, structural features)
    2. Build candidate list using ParetoSelector (prediction only, no I/O)
    3. For each candidate in score order:
       a. Compress + decompress, measure actual ratio and error
       b. If within 10% of target ratio → return immediately
       c. Otherwise adjust parameters via binary search and retry
    4. If no single method meets the target, try cascade composition
    5. Return the best result found
    """

    def __init__(
        self,
        methods: Optional[Dict[str, Any]] = None,
        profiler: Optional[CompressionProfiler] = None,
    ) -> None:
        self._methods: Dict[str, Any] = (
            dict(methods) if methods is not None else dict(METHOD_REGISTRY)
        )
        self._profiler = profiler or CompressionProfiler(
            enable_spectral=True,
            enable_structural=True,
            enable_information=False,
        )
        self._predictor = PredictorRegistry()
        self._searcher = BinarySearchOptimizer(self._methods)
        self._selector = ParetoSelector(self._methods, self._predictor)

    def find_best_method(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        profile: Optional[TensorProfile] = None,
        max_error: float = 0.05,
        max_attempts: int = 3,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Find the best compression method and parameters for *tensor*
        targeting *target_ratio*.

        Parameters
        ----------
        tensor : np.ndarray
            Tensor to compress.
        target_ratio : float
            Desired compression ratio (> 1.0).
        profile : TensorProfile, optional
            Pre-computed profile (computed on demand if None).
        max_error : float
            Maximum acceptable relative error.
        max_attempts : int
            Maximum validation compress/decompress cycles per method.
        validate : bool
            If True, actually compress, decompress, and measure.

        Returns
        -------
        dict
            Result dict with keys:
              - method: selected method name
              - params: parameter dict
              - tier: method tier (1–5)
              - predicted_ratio: predicted compression ratio
              - predicted_error: predicted relative error
              - actual_ratio: measured ratio (if validated)
              - actual_error: measured error (if validated)
              - score: composite selection score
              - candidates: full ranked candidate list (ParetoCandidate dicts)
        """
        if tensor.size == 0:
            return self._empty_result()

        if target_ratio <= 1.0:
            return self._passthrough_result(tensor)

        # Step 1: profile
        if profile is None:
            profile = self._profiler.profile_tensor(tensor)

        # Step 2: select candidates via predictor (fast path)
        candidates = self._selector.select(
            tensor,
            profile,
            target_ratio,
            max_error,
            max_candidates=10,
            pareto_filter=True,
        )

        if not candidates:
            return self._passthrough_result(tensor)

        # Build ranked list of method names by tier (Tier 1 first)
        ranked = self._rank_by_tier(candidates)

        # Step 3: validate top candidates
        best_result: Optional[Dict[str, Any]] = None

        for candidate, tier in ranked:
            method_name = candidate.method_name
            method_instance = self._methods.get(method_name)
            if method_instance is None:
                continue

            params = dict(candidate.params)

            for attempt in range(max_attempts):
                if not validate:
                    # Prediction-only mode
                    result = self._build_result(
                        method_name, params, candidate, profile, tier
                    )
                    return result

                try:
                    actual_ratio, actual_error = self._try_compress(
                        tensor, method_instance, params
                    )
                except Exception:
                    break

                ratio_ok = (
                    abs(actual_ratio - target_ratio) / max(target_ratio, 1.0) <= 0.10
                )
                error_ok = actual_error <= max_error

                result = self._build_result(
                    method_name,
                    params,
                    candidate,
                    profile,
                    tier,
                    actual_ratio,
                    actual_error,
                )

                if best_result is None or (
                    result.get("score", 0.0) > best_result.get("score", 0.0)
                    and (
                        result.get("actual_ratio", 0.0) >= target_ratio * 0.9
                        or best_result.get("actual_ratio", 0.0) < target_ratio * 0.9
                    )
                ):
                    best_result = result

                if ratio_ok and error_ok:
                    return result

                # Adjust parameters and retry
                adjusted = self._adjust_params(
                    method_name, params, actual_ratio, target_ratio
                )
                if adjusted == params:
                    break
                params = adjusted

        # Step 4: fallback — return best found
        if best_result is not None:
            return best_result

        return self._passthrough_result(tensor)

    def _try_compress(
        self,
        tensor: np.ndarray,
        method_instance: Any,
        params: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Run compress + decompress, return (ratio, relative_error)."""
        data, meta = method_instance.compress(tensor, **params)
        ratio = _compute_ratio(tensor.nbytes, data)
        recon = method_instance.decompress(data, meta).reshape(tensor.shape)
        metrics = _compute_metrics(tensor, recon)
        return ratio, metrics["relative_error"]

    def _adjust_params(
        self,
        method_name: str,
        params: Dict[str, Any],
        actual_ratio: float,
        target_ratio: float,
    ) -> Dict[str, Any]:
        """Adjust parameters to bring ratio closer to target."""
        new_params = dict(params)
        ratio_ratio = actual_ratio / max(target_ratio, 1.0)

        if "block_size" in params:
            if ratio_ratio > 1.1:
                new_params["block_size"] = max(params["block_size"] // 2, 2)
            elif ratio_ratio < 0.9:
                new_params["block_size"] = min(params["block_size"] * 2, 4096)
            # Round to even
            new_params["block_size"] = max(int(new_params["block_size"] // 2 * 2), 2)
        elif "rank" in params:
            if ratio_ratio < 0.9:
                new_params["rank"] = max(params["rank"] // 2, 1)
            elif ratio_ratio > 1.1:
                new_params["rank"] = min(params["rank"] * 2, 4096)
        elif "threshold" in params:
            if ratio_ratio < 0.9:
                new_params["threshold"] = min(params["threshold"] * 1.5, 1.0)
            elif ratio_ratio > 1.1:
                new_params["threshold"] = max(params["threshold"] / 1.5, 0.0)
        return new_params

    def _rank_by_tier(
        self, candidates: List[ParetoCandidate]
    ) -> List[Tuple[ParetoCandidate, int]]:
        """Sort candidates: Tier 1–3 first, then by score within each tier."""
        tier1: List[Tuple[ParetoCandidate, int]] = []
        tier5: List[Tuple[ParetoCandidate, int]] = []

        for c in candidates:
            t = c.tier
            if t <= 3:
                tier1.append((c, t))
            else:
                tier5.append((c, t))

        tier1.sort(key=lambda x: -x[0].score)
        tier5.sort(key=lambda x: -x[0].score)

        return tier1 + tier5

    @staticmethod
    def _build_result(
        method_name: str,
        params: Dict[str, Any],
        candidate: ParetoCandidate,
        profile: TensorProfile,
        tier: int,
        actual_ratio: Optional[float] = None,
        actual_error: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "method": method_name,
            "params": params,
            "tier": tier,
            "predicted_ratio": candidate.predicted_ratio,
            "predicted_error": candidate.predicted_error,
            "actual_ratio": actual_ratio,
            "actual_error": actual_error,
            "score": candidate.score,
        }

    def _passthrough_result(self, tensor: np.ndarray) -> Dict[str, Any]:
        return {
            "method": "passthrough",
            "params": {},
            "tier": 0,
            "predicted_ratio": 1.0,
            "predicted_error": 0.0,
            "actual_ratio": 1.0,
            "actual_error": 0.0,
            "score": 0.0,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "method": "none",
            "params": {},
            "tier": 0,
            "predicted_ratio": 1.0,
            "predicted_error": 0.0,
            "score": 0.0,
        }

    def list_methods(self) -> List[Dict[str, Any]]:
        """Describe all registered methods with their tier and category."""
        out: List[Dict[str, Any]] = []
        for name in self._methods:
            tier = int(get_method_tier(name))
            cat = getattr(self._methods[name], "category", "unknown")
            out.append({"name": name, "tier": tier, "category": cat})
        return out

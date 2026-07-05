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
from ..multiplicative_stacking._multiplicativestackingengine import (
    MultiplicativeStackingEngine,
)
from ..target_ratio_engine import PredictorRegistry
from ._stagetype import StageType, _ensure_2d
from .__losslesszstd import _LosslessZstd
from .__losslessrans import _LosslessRANS
from .__svdtruncatedwrapper import _SVDTruncatedWrapper
from .__dct2dwrapper import _DCT2DWrapper
from .__hadamardquantwrapper import _HadamardQuantWrapper


LOSSY_METHOD_REGISTRY = {
    "svd_truncated": _SVDTruncatedWrapper(),
    "dct_2d": _DCT2DWrapper(),
    "hadamard_quant": _HadamardQuantWrapper(),
    "lossless_zstd": _LosslessZstd(),
    "lossless_rans": _LosslessRANS(),
}

STAGE_METHOD_MAP = {
    StageType.DECOMPOSITION: [("svd_truncated", "rank", {"rank": 16})],
    StageType.SPECTRAL: [("dct_2d", "keep_fraction", {"keep_fraction": 0.1})],
    StageType.QUANTIZATION: [
        ("hadamard_quant", "block_size", {"block_size": 64, "bits": 4})
    ],
    StageType.ENTROPY: [("lossless_zstd", "level", {"level": 3})],
}

STAGE_MIN_RATIO = {
    StageType.DECOMPOSITION: 2.0,
    StageType.SPECTRAL: 1.5,
    StageType.QUANTIZATION: 2.0,
    StageType.ENTROPY: 1.2,
}

STAGE_MAX_RATIO = {
    StageType.DECOMPOSITION: 100.0,
    StageType.SPECTRAL: 50.0,
    StageType.QUANTIZATION: 20.0,
    StageType.ENTROPY: 5.0,
}

lagrangian_allocate_sub_ratios = (
    MultiplicativeStackingEngine.lagrangian_allocate_sub_ratios
)
predict_svd_error = PredictorRegistry.predict_svd_error
predict_block_quant_error = PredictorRegistry.predict_block_quant_error


class CascadeComposer:
    """Composes compression methods into a residual cascade.

    Wraps ``MultiplicativeStackingEngine`` from *multiplicative_stacking*
    for the actual stacking work.  This file keeps backward compatibility
    with code that imports ``CascadeComposer`` directly.
    """

    def __init__(
        self,
        registry: Optional[Dict[str, Any]] = None,
        profile: Optional[TensorProfile] = None,
    ) -> None:
        self.registry = registry or dict(LOSSY_METHOD_REGISTRY)
        self.registry["svd_truncated"] = _SVDTruncatedWrapper()
        self.registry["dct_2d"] = _DCT2DWrapper()
        self.registry["hadamard_quant"] = _HadamardQuantWrapper()
        self.profile = profile
        self.searcher = ParameterSearcher(profile) if profile else None

        # Build a minimal engine for MultiplicativeStackingEngine
        self._stacking_engine: Optional[Any] = None

    def _get_stacking_engine(self):
        if self._stacking_engine is None:
            from ..multiplicative_stacking import MultiplicativeStackingEngine

            class _MinimalStackingEngine:
                """Minimal stub that exposes _methods for MultiplicativeStackingEngine."""

                def __init__(self, registry):
                    self._methods = dict(registry)

            self._stacking_engine = MultiplicativeStackingEngine(
                _MinimalStackingEngine(self.registry)
            )
        return self._stacking_engine

    def compose(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        error_budget: float,
        profile: Optional[TensorProfile] = None,
        preferred_pattern: Optional[List[StageType]] = None,
    ) -> List[CascadeStageConfig]:
        if profile is not None:
            self.profile = profile
            self.searcher = ParameterSearcher(profile)
        if self.profile is None:
            self.profile = TensorProfile()
            self.searcher = ParameterSearcher(self.profile)

        if preferred_pattern is not None:
            patterns_to_try = [preferred_pattern]
        else:
            patterns_to_try = self._select_patterns(tensor, target_ratio)

        best_stages: Optional[List[CascadeStageConfig]] = None
        best_score = float("inf")

        for pattern in patterns_to_try:
            try:
                stages = self._instantiate_pattern(tensor, target_ratio, pattern)
                stages = lagrangian_allocate_sub_ratios(
                    target_ratio, error_budget, stages, self.profile
                )
                stages = self._tune_stages(tensor, stages)

                total_error = sum(s.predicted_error for s in stages)
                actual_product = 1.0
                for s in stages:
                    actual_product *= max(s.sub_target_ratio, 1.0)

                ratio_shortfall = max(target_ratio / max(actual_product, 1), 1.0)
                if ratio_shortfall <= 1.1:
                    penalty = 0.0
                elif ratio_shortfall <= 2.0:
                    penalty = 0.2 * (ratio_shortfall - 1.1)
                else:
                    penalty = 0.5 + 0.1 * (ratio_shortfall - 2.0)

                has_passthrough = any(
                    s.sub_target_ratio < 1.1 and s.stage_type != StageType.ENTROPY
                    for s in stages
                )
                if has_passthrough:
                    penalty += 10.0

                score = total_error + penalty
                if score < best_score:
                    best_score = score
                    best_stages = stages
            except Exception:
                continue

        if best_stages is None:
            best_stages = self._fallback_cascade(tensor, target_ratio)

        return best_stages

    def _select_patterns(
        self,
        tensor: np.ndarray,
        target_ratio: float,
    ) -> List[List[StageType]]:
        can_decompose = tensor.ndim >= 2 and min(tensor.shape) >= 4
        patterns: List[List[StageType]] = []

        if target_ratio <= 8:
            if can_decompose:
                patterns.append([StageType.DECOMPOSITION, StageType.QUANTIZATION])
            else:
                patterns.append([StageType.SPECTRAL, StageType.QUANTIZATION])
        elif target_ratio <= 30:
            if can_decompose:
                patterns.append([StageType.DECOMPOSITION, StageType.QUANTIZATION])
            patterns.append([StageType.SPECTRAL, StageType.QUANTIZATION])
        elif target_ratio <= 200:
            if can_decompose:
                patterns.append(
                    [StageType.DECOMPOSITION, StageType.QUANTIZATION, StageType.ENTROPY]
                )
            patterns.append(
                [StageType.SPECTRAL, StageType.QUANTIZATION, StageType.ENTROPY]
            )
            if can_decompose:
                patterns.append(
                    [
                        StageType.DECOMPOSITION,
                        StageType.SPECTRAL,
                        StageType.QUANTIZATION,
                        StageType.ENTROPY,
                    ]
                )
        else:
            if can_decompose:
                patterns.append(
                    [StageType.DECOMPOSITION, StageType.QUANTIZATION, StageType.ENTROPY]
                )
                patterns.append(
                    [
                        StageType.DECOMPOSITION,
                        StageType.SPECTRAL,
                        StageType.QUANTIZATION,
                        StageType.ENTROPY,
                    ]
                )
            patterns.append(
                [StageType.SPECTRAL, StageType.QUANTIZATION, StageType.ENTROPY]
            )

        if not patterns:
            patterns.append([StageType.QUANTIZATION])
        return patterns

    def _instantiate_pattern(
        self,
        tensor: np.ndarray,
        target_ratio: float,
        pattern: List[StageType],
    ) -> List[CascadeStageConfig]:
        stages: List[CascadeStageConfig] = []
        tensor_2d = _ensure_2d(tensor)

        min_r = [STAGE_MIN_RATIO.get(s, 1.1) for s in pattern]
        product_min = 1.0
        for r in min_r:
            product_min *= r
        effective_target = max(target_ratio, product_min)
        sub_ratio = effective_target ** (1.0 / len(pattern))

        for stage_type in pattern:
            method_choices = STAGE_METHOD_MAP.get(stage_type, [])
            if not method_choices:
                continue
            method_name, param_key, default_params = method_choices[0]
            params = dict(default_params)
            max_r = STAGE_MAX_RATIO.get(stage_type, 100.0)
            capped_sub = min(sub_ratio, max_r)

            if stage_type == StageType.DECOMPOSITION and self.searcher:
                m, n = tensor_2d.shape
                needed_rank = max(1, int(m * n / (capped_sub * (m + n))))
                rank = min(needed_rank, min(m, n) - 1)
                params["rank"] = max(rank, 1)

            elif stage_type == StageType.SPECTRAL and self.searcher:
                if method_name == "dct_2d":
                    kf = self.searcher.find_keep_fraction_for_target(
                        tensor_2d.shape, capped_sub
                    )
                    params["keep_fraction"] = max(kf, 0.01)
                elif method_name in ("hadamard_int4", "hadamard_int8"):
                    bits = 4 if "int4" in method_name else 8
                    bs = self.searcher.find_block_size_for_target(
                        tensor_2d.size, capped_sub, bits
                    )
                    params["block_size"] = bs

            elif stage_type == StageType.QUANTIZATION and self.searcher:
                bits = 4 if "int4" in method_name else 8
                bs = self.searcher.find_block_size_for_target(
                    tensor_2d.size, capped_sub, bits
                )
                params["block_size"] = bs

            elif stage_type == StageType.ENTROPY:
                params = {"level": 3}

            stages.append(
                CascadeStageConfig(
                    stage_type=stage_type,
                    method_name=method_name,
                    params=params,
                    sub_target_ratio=capped_sub,
                )
            )
        return stages

    def _tune_stages(
        self,
        tensor: np.ndarray,
        stages: List[CascadeStageConfig],
    ) -> List[CascadeStageConfig]:
        tuned: List[CascadeStageConfig] = []
        original_flat = tensor.astype(np.float64).ravel()
        residual = original_flat.copy()

        for i, stage in enumerate(stages):
            if stage.stage_type == StageType.ENTROPY:
                stage.predicted_error = 0.0
                tuned.append(stage)
                continue

            method = self.registry.get(stage.method_name)
            if method is None:
                tuned.append(stage)
                continue

            try:
                stage_input = residual.astype(np.float32).reshape(tensor.shape)
                tensor_2d = _ensure_2d(stage_input)
                cd, meta = method.compress(tensor_2d, **stage.params)

                recon = method.decompress(cd, meta).reshape(tensor.shape)
                stage.sub_target_ratio = _compute_ratio(tensor.nbytes, cd)

                reconstructed = recon.astype(np.float64).ravel()
                residual -= reconstructed

                stage.predicted_error = float(
                    np.linalg.norm(residual) / (np.linalg.norm(original_flat) + 1e-30)
                )
            except Exception:
                if self.profile:
                    if stage.stage_type == StageType.DECOMPOSITION:
                        stage.predicted_error = predict_svd_error(
                            self.profile, stage.params.get("rank", 16)
                        )
                    elif stage.stage_type == StageType.SPECTRAL:
                        stage.predicted_error = predict_block_quant_error(
                            self.profile, stage.params.get("block_size", 64), 4
                        )
                    elif stage.stage_type == StageType.QUANTIZATION:
                        stage.predicted_error = predict_block_quant_error(
                            self.profile, stage.params.get("block_size", 32), 4
                        )
                    else:
                        stage.predicted_error = 0.01
            tuned.append(stage)
        return tuned

    def _fallback_cascade(
        self,
        tensor: np.ndarray,
        target_ratio: float,
    ) -> List[CascadeStageConfig]:
        tensor_2d = _ensure_2d(tensor)
        m, n = tensor_2d.shape
        stages: List[CascadeStageConfig] = []

        svd_ratio = STAGE_MIN_RATIO[StageType.DECOMPOSITION]
        quant_ratio = STAGE_MIN_RATIO[StageType.QUANTIZATION]
        entropy_ratio = STAGE_MIN_RATIO[StageType.ENTROPY]

        if m >= 4 and n >= 4:
            rank = max(1, int(m * n / (target_ratio * (m + n) * 2)))
            rank = min(rank, min(m, n) - 1)
            stages.append(
                CascadeStageConfig(
                    stage_type=StageType.DECOMPOSITION,
                    method_name="svd_truncated",
                    params={"rank": max(rank, 1)},
                    sub_target_ratio=svd_ratio,
                )
            )

        block_size = max(2, min(64, int(m * n / (target_ratio * 4))))
        stages.append(
            CascadeStageConfig(
                stage_type=StageType.QUANTIZATION,
                method_name="block_int4",
                params={"block_size": block_size},
                sub_target_ratio=quant_ratio,
            )
        )

        remaining = target_ratio / (svd_ratio * quant_ratio)
        if remaining > 1.5:
            stages.append(
                CascadeStageConfig(
                    stage_type=StageType.ENTROPY,
                    method_name="lossless_zstd",
                    params={"level": 3},
                    sub_target_ratio=entropy_ratio,
                )
            )
        return stages

    def execute_cascade(
        self,
        tensor: np.ndarray,
        stages: List[CascadeStageConfig],
        return_intermediate: bool = False,
    ) -> Tuple[np.ndarray, float, float, List[Tuple[str, float, float]]]:
        # Delegate to MultiplicativeStackingEngine for the actual stacking
        stacking = self._get_stacking_engine()
        plan = stacking.plan_stacking(
            tensor, tensor_name="", target_ratio=100.0, max_error=0.01
        )
        if plan.stages:
            data, meta = stacking.execute_stacking(plan, tensor)
            recon = stacking.unstack(data, meta, tensor.shape)
            ratio = tensor.nbytes / max(len(data), 1)
            error = float(
                np.linalg.norm(
                    tensor.ravel().astype(np.float64) - recon.ravel().astype(np.float64)
                )
                / (np.linalg.norm(tensor.ravel().astype(np.float64)) + 1e-30)
            )
            stage_results = [
                (s.method_name, s.sub_ratio, s.sub_error) for s in plan.stages
            ]
            if return_intermediate:
                return recon, ratio, error, stage_results
            return recon, ratio, error, stage_results
        return tensor, 1.0, 0.0, []

    def validate_cascade(
        self,
        original: np.ndarray,
        stages: List[CascadeStageConfig],
    ) -> Dict[str, Any]:
        _, actual_ratio, actual_error, stage_results = self.execute_cascade(
            original, stages, return_intermediate=True
        )
        return {
            "actual_ratio": actual_ratio,
            "actual_error": actual_error,
            "n_stages": len(stages),
            "stage_results": stage_results,
            "stage_methods": [s.method_name for s in stages],
            "stage_params": [s.params for s in stages],
            "stage_sub_targets": [s.sub_target_ratio for s in stages],
            "predicted_errors": [s.predicted_error for s in stages],
        }

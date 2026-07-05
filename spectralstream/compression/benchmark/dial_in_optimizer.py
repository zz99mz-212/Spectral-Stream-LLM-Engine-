from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine import CompressionIntelligenceEngine
from spectralstream.compression.benchmark.loss_calculator import (
    LossCalculator,
    TensorLossMetrics,
)

logger = logging.getLogger(__name__)

DEFAULT_PARAM_GRID: Dict[str, Dict[str, List[Any]]] = {
    "block_int8": {"block_size": [32, 64, 128, 256, 512]},
    "block_int4": {"block_size": [32, 64, 128, 256, 512]},
    "hadamard_int8": {"block_size": [64, 128, 256, 512]},
    "hadamard_int4": {"block_size": [64, 128, 256, 512]},
    "delta_int4": {"block_size": [32, 64, 128, 256], "group_size": [16, 32, 64]},
    "sparsity_int4": {"block_size": [32, 64, 128]},
    "svd_compress": {"rank": [4, 8, 16, 32, 64, 128]},
    "dct_spectral": {"keep_ratio": [0.01, 0.05, 0.1, 0.2, 0.5]},
    "tensor_train": {"rank": [4, 8, 16, 32]},
    "fwht_compress": {"keep_ratio": [0.05, 0.1, 0.2, 0.5]},
}

TENSOR_TYPE_ORDER = [
    "embedding",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "output",
    "norm",
    "weight",
]


@dataclass
class ParamGridResult:
    method: str
    tensor_type: str
    best_params: Dict[str, Any]
    best_error: float
    best_ratio: float
    best_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CompressionRecipe:
    tensor_type_recipes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_method: str = "block_int8"
    default_params: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    num_tensor_types: int = 0


class DialInOptimizer:
    def __init__(
        self,
        engine: Optional[CompressionIntelligenceEngine] = None,
        param_grid: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    ):
        self._engine = engine or CompressionIntelligenceEngine()
        self._loss = LossCalculator()
        self._param_grid = param_grid or DEFAULT_PARAM_GRID

    @property
    def engine(self) -> CompressionIntelligenceEngine:
        return self._engine

    def optimize_method(
        self,
        method_name: str,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        target_ratio: float = 100.0,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> ParamGridResult:
        inst = self._engine._methods.get(method_name)
        if inst is None:
            raise ValueError(f"Method '{method_name}' not found in engine")

        grid = param_grid or self._param_grid.get(method_name, {})
        if not grid:
            return ParamGridResult(
                method=method_name,
                tensor_type=tensor_type,
                best_params={},
                best_error=1.0,
                best_ratio=1.0,
                best_score=0.0,
                all_results=[],
            )

        all_results: List[Dict[str, Any]] = []
        best_score = -1.0
        best_params: Dict[str, Any] = {}
        best_error = 1.0
        best_ratio = 1.0

        keys = list(grid.keys())
        values = list(grid.values())

        from itertools import product

        for combo in product(*values):
            params = dict(zip(keys, combo))
            try:
                data, meta = inst.compress(tensor, **params)
                recon = inst.decompress(data, meta)
                if recon.shape != tensor.shape:
                    recon = recon.reshape(tensor.shape)

                metrics = self._loss.compute_tensor_metrics(
                    tensor, recon, tensor.nbytes, len(data)
                )
                score = self._composite_optimization_score(metrics, target_ratio)

                all_results.append(
                    {
                        "params": params,
                        "ratio": metrics.compression_ratio,
                        "error": metrics.relative_error,
                        "snr": metrics.snr_db,
                        "cosine": metrics.cosine_similarity,
                        "outlier_preservation": metrics.outlier_preservation_ratio,
                        "bit_precision": metrics.bit_precision_achieved,
                        "score": score,
                    }
                )

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_error = metrics.relative_error
                    best_ratio = metrics.compression_ratio

            except Exception as e:
                logger.debug("  params=%s failed: %s", params, e)
                continue

        all_results.sort(key=lambda x: -x["score"])

        return ParamGridResult(
            method=method_name,
            tensor_type=tensor_type,
            best_params=best_params,
            best_error=best_error,
            best_ratio=best_ratio,
            best_score=best_score,
            all_results=all_results[:20],
        )

    def optimize_tensor_type(
        self,
        tensor_type: str,
        target_ratio: float = 100.0,
        seed: int = 42,
    ) -> List[ParamGridResult]:
        from spectralstream.compression.benchmark.benchmark_runner import (
            SYNTHETIC_SHAPES,
        )

        shapes = SYNTHETIC_SHAPES.get(tensor_type, [(4096, 4096)])
        rng = np.random.RandomState(seed)
        shape = shapes[0]
        tensor = rng.randn(*shape).astype(np.float32)
        if tensor_type == "norm":
            tensor = np.abs(tensor) * 0.1

        results: List[ParamGridResult] = []
        for method_name in self._param_grid:
            try:
                result = self.optimize_method(
                    method_name, tensor, tensor_type, target_ratio
                )
                if result.all_results:
                    results.append(result)
            except ValueError:
                continue
            except Exception as e:
                logger.debug("  %s on %s failed: %s", method_name, tensor_type, e)

        results.sort(key=lambda x: -x.best_score)
        return results

    def optimize_all_tensor_types(
        self,
        target_ratio: float = 100.0,
        tensor_types: Optional[List[str]] = None,
    ) -> Dict[str, List[ParamGridResult]]:
        types_to_test = tensor_types or TENSOR_TYPE_ORDER
        results: Dict[str, List[ParamGridResult]] = {}

        for ttype in types_to_test:
            logger.info("Optimizing for tensor type: %s", ttype)
            try:
                t_results = self.optimize_tensor_type(ttype, target_ratio)
                if t_results:
                    results[ttype] = t_results
            except Exception as e:
                logger.warning("  Optimization failed for %s: %s", ttype, e)

        return results

    def generate_recipe(
        self,
        target_ratio: float = 100.0,
        tensor_types: Optional[List[str]] = None,
        results: Optional[Dict[str, List[ParamGridResult]]] = None,
    ) -> CompressionRecipe:
        if results is None:
            results = self.optimize_all_tensor_types(target_ratio, tensor_types)

        recipe: Dict[str, Dict[str, Any]] = {}
        for ttype, t_results in results.items():
            if t_results:
                best = t_results[0]
                recipe[ttype] = {
                    "method": best.method,
                    "params": best.best_params,
                    "expected_error": round(best.best_error, 6),
                    "expected_ratio": round(best.best_ratio, 2),
                    "score": round(best.best_score, 4),
                }
            else:
                recipe[ttype] = {
                    "method": "block_int8",
                    "params": {"block_size": 128},
                    "expected_error": 0.01,
                    "expected_ratio": 4.0,
                    "score": 0.0,
                }

        from datetime import datetime

        return CompressionRecipe(
            tensor_type_recipes=recipe,
            default_method="block_int8",
            default_params={"block_size": 128},
            generated_at=datetime.now().isoformat(),
            num_tensor_types=len(recipe),
        )

    def save_recipe(self, recipe: CompressionRecipe, path: str) -> str:
        data = {
            "tensor_type_recipes": recipe.tensor_type_recipes,
            "default_method": recipe.default_method,
            "default_params": recipe.default_params,
            "generated_at": recipe.generated_at,
            "num_tensor_types": recipe.num_tensor_types,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Recipe saved to %s", path)
        return path

    def load_recipe(self, path: str) -> CompressionRecipe:
        with open(path) as f:
            data = json.load(f)
        return CompressionRecipe(
            tensor_type_recipes=data.get("tensor_type_recipes", {}),
            default_method=data.get("default_method", "block_int8"),
            default_params=data.get("default_params", {"block_size": 128}),
            generated_at=data.get("generated_at", ""),
            num_tensor_types=data.get("num_tensor_types", 0),
        )

    def _composite_optimization_score(
        self, m: TensorLossMetrics, target_ratio: float
    ) -> float:
        rel = max(0.0, 1.0 - min(m.relative_error * 20, 10.0))
        snr = min(1.0, max(0.0, m.snr_db / 60.0)) if m.snr_db != float("inf") else 1.0
        cos = max(0.0, (m.cosine_similarity + 1.0) / 2.0)
        ratio_factor = min(m.compression_ratio / target_ratio, 2.0) / 2.0
        outlier = max(0.0, m.outlier_preservation_ratio)
        bits = m.bit_precision_achieved / 32.0

        return float(
            ratio_factor * 0.25
            + rel * 0.15
            + snr * 0.10
            + cos * 0.10
            + outlier * 0.15
            + bits * 0.15
        )

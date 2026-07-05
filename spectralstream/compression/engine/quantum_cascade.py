"""
Quantum Parallel Cascade — test compression methods in quantum-inspired
superposition (parallel) and collapse to the best result.

Contains two engine variants:
- QuantumCascadeEngine (legacy): parallel method testing with confidence-based
  early stopping and CascadeResult/CascadeReport dataclasses.
- QuantumSuperpositionEngine (v2): full quantum-inspired superposition testing
  with QuantumSuperpositionTest, CascadeSuperpositionPlan, and multi-stage
  cascade execution.

Both use ThreadPoolExecutor for parallel execution. The v2 engine adds:
- Composite scoring (ratio reward × error penalty)
- Multi-stage cascade planning via CascadeSuperpositionPlan.build_for_target()
- Memory-adaptive parallelism
- Full metrics (SNR, cosine similarity)
"""

from __future__ import annotations

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy Dataclasses and Engine
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CascadeResult:
    """Result of testing a single compression method."""

    method_name: str
    method_params: Dict[str, Any]
    compressed_data: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None
    ratio: float = 0.0
    error: float = 1.0
    score: float = 0.0
    time_seconds: float = 0.0
    success: bool = False


@dataclass
class CascadeReport:
    """Report from a quantum cascade operation."""

    results: List[CascadeResult] = field(default_factory=list)
    best_result: Optional[CascadeResult] = None
    n_tested: int = 0
    n_success: int = 0
    total_time: float = 0.0
    mode: str = "balanced"
    target_ratio: float = 5000.0
    max_error: float = 0.01
    early_stopped: bool = False


class QuantumCascadeEngine:
    """Parallel method testing engine with confidence-based early stopping.

    Tests multiple compression methods in parallel, ranks them by
    score = ratio / (1 + error * penalty), and supports early stopping
    when a method achieves the target with acceptable error.

    Usage::

        engine = QuantumCascadeEngine(max_workers=4)
        report = engine.test_methods(
            tensor=tensor,
            methods=[...],
            target_ratio=5000.0,
            max_error=0.01,
            mode="balanced",
        )
        best = report.best_result  # CascadeResult with highest score
    """

    ERROR_PENALTY = 10.0

    MODE_LIMITS = {
        "fast": 3,
        "balanced": 10,
        "extreme": 200,
    }

    EARLY_STOP_CONFIDENCE = {
        "fast": 0.3,
        "balanced": 0.7,
        "extreme": 0.95,
    }

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._n_total_tested = 0
        self._n_total_success = 0
        self._total_test_time = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_tested": self._n_total_tested,
            "total_success": self._n_total_success,
            "total_time": self._total_test_time,
            "max_workers": self._max_workers,
        }

    def test_methods(
        self,
        tensor: np.ndarray,
        methods: List[Dict[str, Any]],
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        mode: str = "balanced",
        profile: Any = None,
        error_budget: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> CascadeReport:
        t0 = time.perf_counter()
        eb = (
            error_budget
            if error_budget is not None
            else max_error / max(target_ratio, 1.0)
        )

        limit = self.MODE_LIMITS.get(mode, 10)
        candidates = methods[:limit]

        if not candidates:
            return CascadeReport(
                mode=mode, target_ratio=target_ratio, max_error=max_error
            )

        n_total = len(candidates)
        results: List[CascadeResult] = []
        early_stopped = False

        if self._max_workers > 1 and len(candidates) >= 2:
            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(candidates))
            ) as pool:
                future_map = {}
                for mdict in candidates:
                    future = pool.submit(
                        self._test_single_method,
                        tensor=tensor,
                        mdict=mdict,
                        tensor_name="",
                        error_budget=eb,
                    )
                    future_map[future] = mdict

                for future in as_completed(future_map):
                    mdict = future_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result.success:
                            if (
                                result.ratio >= target_ratio
                                and result.error <= max_error
                            ):
                                threshold = self.EARLY_STOP_CONFIDENCE.get(mode, 0.7)
                                if result.score >= threshold:
                                    early_stopped = True
                                    for f in future_map:
                                        if not f.done():
                                            f.cancel()
                                    break
                    except Exception as exc:
                        logger.debug("Method test failed: %s", exc)
                        results.append(
                            CascadeResult(
                                method_name=str(mdict.get("name", "unknown")),
                                method_params=mdict.get("params", {}),
                                success=False,
                            )
                        )

                    if progress_callback:
                        progress_callback(len(results), n_total)
        else:
            for mdict in candidates:
                try:
                    result = self._test_single_method(
                        tensor=tensor,
                        mdict=mdict,
                        tensor_name="",
                        error_budget=eb,
                    )
                    results.append(result)

                    if (
                        result.success
                        and result.ratio >= target_ratio
                        and result.error <= max_error
                    ):
                        threshold = self.EARLY_STOP_CONFIDENCE.get(mode, 0.7)
                        if result.score >= threshold:
                            early_stopped = True
                            break
                except Exception as exc:
                    logger.debug("Sequential method test failed: %s", exc)
                    results.append(
                        CascadeResult(
                            method_name=str(mdict.get("name", "unknown")),
                            method_params=mdict.get("params", {}),
                            success=False,
                        )
                    )

                if progress_callback:
                    progress_callback(len(results), n_total)

        results.sort(key=lambda r: -r.score)
        best = results[0] if results and results[0].success else None

        elapsed = time.perf_counter() - t0

        self._n_total_tested += len(results)
        self._n_total_success += sum(1 for r in results if r.success)
        self._total_test_time += elapsed

        return CascadeReport(
            results=results,
            best_result=best,
            n_tested=len(results),
            n_success=sum(1 for r in results if r.success),
            total_time=elapsed,
            mode=mode,
            target_ratio=target_ratio,
            max_error=max_error,
            early_stopped=early_stopped,
        )

    def _test_single_method(
        self,
        tensor: np.ndarray,
        mdict: Dict[str, Any],
        tensor_name: str = "",
        error_budget: float = 0.01,
    ) -> CascadeResult:
        method_name = mdict.get("name", "unknown")
        instance = mdict.get("instance")
        params = mdict.get("params", {})

        t1 = time.perf_counter()
        try:
            if instance is None:
                raise ValueError(f"No instance for method '{method_name}'")

            if hasattr(instance, "compress"):
                data, meta = instance.compress(tensor, **params)
            else:
                data, meta = instance(tensor, **params)

            if data is None or len(data) == 0:
                raise ValueError("No compressed data produced")

            ratio = tensor.nbytes / max(len(data), 1)
            error = _estimate_error(tensor, data, meta, instance)

            score = ratio / (1.0 + error * self.ERROR_PENALTY)

            elapsed = time.perf_counter() - t1

            return CascadeResult(
                method_name=method_name,
                method_params=params,
                compressed_data=data,
                metadata=meta,
                ratio=ratio,
                error=error,
                score=score,
                time_seconds=elapsed,
                success=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t1
            logger.debug("Method '%s' failed in %.2fs: %s", method_name, elapsed, exc)
            return CascadeResult(
                method_name=method_name,
                method_params=params,
                success=False,
                time_seconds=elapsed,
            )

    def test_on_representative(
        self,
        tensor: np.ndarray,
        engine: Any,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        mode: str = "balanced",
        name: str = "",
    ) -> CascadeReport:
        all_methods = []
        for mname, minst in engine._methods.items():
            all_methods.append(
                {
                    "name": mname,
                    "instance": minst,
                    "params": {},
                }
            )

        profile = engine.profile_tensor(tensor, name)

        return self.test_methods(
            tensor=tensor,
            methods=all_methods,
            target_ratio=target_ratio,
            max_error=max_error,
            mode=mode,
            profile=profile,
            error_budget=max_error / max(target_ratio, 1.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# V2: Quantum Superposition Engine
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QuantumSuperpositionTest:
    """Result of testing a method in quantum-inspired superposition."""

    method_name: str
    instance: Any
    params: Dict[str, Any] = field(default_factory=dict)
    compressed_data: Optional[bytes] = None
    decompressed: Optional[np.ndarray] = None
    ratio: float = 0.0
    error: float = 0.0
    snr_db: float = 0.0
    cosine_similarity: float = 0.0
    time_seconds: float = 0.0
    success: bool = False
    exception: Optional[str] = None

    def composite_score(
        self, target_ratio: float = 5000.0, max_error: float = 0.01
    ) -> float:
        """Compute composite score: ratio reward × error penalty.

        Higher is better. Methods that exceed target_ratio and
        stay under max_error get the highest scores.
        """
        if not self.success:
            return -1.0
        ratio_reward = min(self.ratio / target_ratio, 2.0)
        error_penalty = max(0.0, 1.0 - self.error / max(max_error, 1e-10))
        return ratio_reward * error_penalty


@dataclass
class CascadeStage:
    """A single stage in a multi-stage cascade."""

    methods: List[Any]
    target_ratio: float
    max_error: float


@dataclass
class CascadeSuperpositionPlan:
    """Multi-stage cascade where each stage tests methods in superposition.

    For extreme ratios (5000:1), a single method may not suffice.
    This plan chains multiple stages: Stage 1 (decomposition) →
    Stage 2 (spectral) → Stage 3 (structural) → Stage 4 (quantization).
    Each stage tests its methods in parallel superposition.
    """

    stages: List[CascadeStage]
    overall_target: float
    overall_max_error: float

    @classmethod
    def build_for_target(
        cls,
        target_ratio: float,
        available_methods: List[Any],
    ) -> CascadeSuperpositionPlan:
        """Build a cascade plan optimized for the target ratio.

        Uses the tier system to assign methods to stages:
        - Tier 1-2 (decomposition/spectral): Stage 1
        - Tier 3 (structural): Stage 2
        - Tier 4 (entropy): Stage 3
        - Tier 5 (quantization): Stage 4 (last resort)

        Allocates sub-targets using geometric progression.
        """
        tier_stages: Dict[int, List[Any]] = {1: [], 2: [], 3: [], 4: [], 5: []}
        for m in available_methods:
            if isinstance(m, dict):
                tier = m.get("tier", 5)
            else:
                tier = getattr(m, "tier", 5)
            tier_val = tier.value if hasattr(tier, "value") else int(tier)
            tier_stages.setdefault(tier_val, []).append(m)

        n_stages = sum(1 for t in [1, 2, 3, 4, 5] if tier_stages.get(t))
        n_stages = max(n_stages, 1)

        sub_target = target_ratio ** (1.0 / n_stages)
        sub_error = 1.0 - (1.0 - 0.01) ** (1.0 / n_stages)

        stages: List[CascadeStage] = []
        for t in [1, 2, 3, 4, 5]:
            methods = tier_stages.get(t, [])
            if not methods:
                continue
            stages.append(
                CascadeStage(
                    methods=methods,
                    target_ratio=sub_target,
                    max_error=sub_error,
                )
            )

        return cls(
            stages=stages,
            overall_target=target_ratio,
            overall_max_error=sub_error * n_stages,
        )


class QuantumSuperpositionEngine:
    """Tests compression methods in quantum-inspired superposition.

    Instead of testing methods sequentially, launches all of them
    in parallel using ThreadPoolExecutor, then selects the best
    result. Conceptually similar to quantum superposition where
    all states exist simultaneously until measurement.

    Key optimizations:
    - Parallel execution: all methods tested concurrently
    - Early termination: if a method exceeds target with low error,
      remaining pending futures can be cancelled
    - Adaptive batch size: based on available CPU cores
    - Memory-aware: limits parallelism for large tensors
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        early_termination: bool = True,
    ):
        self._max_workers = max_workers or os.cpu_count() or 4
        self._early_termination = early_termination

    def _compute_adaptive_parallelism(
        self,
        tensor_nbytes: int,
        n_candidates: int,
        memory_budget_gb: float = 48.0,
    ) -> int:
        """Determine optimal parallelism level."""
        memory_bytes = memory_budget_gb * (1024**3)
        per_method_bytes = tensor_nbytes * 3
        max_by_memory = max(1, int(memory_bytes / max(per_method_bytes, 1)))
        max_by_cpu = self._max_workers
        return min(max_by_memory, max_by_cpu, n_candidates)

    def _test_single_method(
        self,
        tensor: np.ndarray,
        candidate: Any,
        target_ratio: float,
        max_error: float,
    ) -> QuantumSuperpositionTest:
        """Compress+decompress+measure a single method."""
        if isinstance(candidate, dict):
            inst = candidate.get("instance")
            mname = candidate.get("name", getattr(inst, "name", "unknown"))
            params = candidate.get("params", {})
        else:
            inst = getattr(candidate, "instance", candidate)
            mname = getattr(candidate, "name", getattr(inst, "name", "unknown"))
            params = getattr(candidate, "params", {})

        result = QuantumSuperpositionTest(
            method_name=mname,
            instance=inst,
            params=params,
        )

        t0 = time.perf_counter()
        try:
            if hasattr(inst, "compress"):
                data, meta = inst.compress(tensor, **params)
            else:
                result.exception = f"{mname}: no compress method"
                result.time_seconds = time.perf_counter() - t0
                return result
        except Exception as exc:
            result.exception = f"{mname} compress failed: {exc}"
            result.time_seconds = time.perf_counter() - t0
            return result

        try:
            if hasattr(inst, "decompress"):
                recon = inst.decompress(data, meta)
            else:
                result.exception = f"{mname}: no decompress method"
                result.time_seconds = time.perf_counter() - t0
                return result
        except Exception as exc:
            result.exception = f"{mname} decompress failed: {exc}"
            result.time_seconds = time.perf_counter() - t0
            return result

        if recon.shape != tensor.shape:
            try:
                recon = recon.reshape(tensor.shape)
            except Exception as exc:
                result.exception = (
                    f"{mname} shape mismatch: {recon.shape} vs {tensor.shape}: {exc}"
                )
                result.time_seconds = time.perf_counter() - t0
                return result

        var = float(np.var(tensor))
        mse = float(np.mean((tensor.ravel() - recon.ravel()) ** 2))
        relative_error = mse / var if var > 1e-30 else float(mse)
        snr_db = 10.0 * np.log10(var / mse) if mse > 1e-30 else 100.0
        dot = float(np.dot(tensor.ravel(), recon.ravel()))
        norm_p = float(np.linalg.norm(tensor.ravel()))
        norm_q = float(np.linalg.norm(recon.ravel()))
        cos_sim = dot / max(norm_p * norm_q, 1e-30)

        ratio = tensor.nbytes / max(len(data), 1)

        result.compressed_data = data
        result.decompressed = recon
        result.ratio = ratio
        result.error = relative_error
        result.snr_db = snr_db
        result.cosine_similarity = cos_sim
        result.success = True
        result.time_seconds = time.perf_counter() - t0

        return result

    def test_in_superposition(
        self,
        tensor: np.ndarray,
        candidates: List[Any],
        target_ratio: float,
        max_error: float,
        tensor_nbytes: int,
        memory_budget_gb: float = 48.0,
    ) -> Tuple[QuantumSuperpositionTest, List[QuantumSuperpositionTest]]:
        """Test all candidate methods in parallel superposition.

        Flow:
        1. Determine optimal degree of parallelism based on:
           - Number of CPU cores
           - Tensor size (large tensors = fewer parallel methods)
           - Available memory budget
        2. Launch method tests in ThreadPoolExecutor batches
        3. As each completes, evaluate composite_score
        4. Early termination: if any method meets both ratio AND
           error targets, cancel remaining pending futures
        5. Collapse superposition: return best result + all results

        Returns:
            (best_result, all_results)
        """
        batch_size = self._compute_adaptive_parallelism(
            tensor_nbytes, len(candidates), memory_budget_gb
        )

        all_results: List[QuantumSuperpositionTest] = []
        best_result: Optional[QuantumSuperpositionTest] = None
        best_score: float = -1.0

        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            futures: List[Future] = []
            executor = ThreadPoolExecutor(max_workers=batch_size)

            for candidate in batch:
                future = executor.submit(
                    self._test_single_method,
                    tensor,
                    candidate,
                    target_ratio,
                    max_error,
                )
                futures.append(future)

            try:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)

                        if result.success:
                            score = result.composite_score(target_ratio, max_error)
                            if score > best_score:
                                best_score = score
                                best_result = result

                            if (
                                self._early_termination
                                and result.ratio >= target_ratio
                                and result.error <= max_error
                            ):
                                for f in futures:
                                    f.cancel()
                                logger.info(
                                    "Early termination: %s met targets "
                                    "(ratio=%.1f, error=%.6f)",
                                    result.method_name,
                                    result.ratio,
                                    result.error,
                                )
                                break
                    except Exception:
                        continue
            finally:
                executor.shutdown(wait=False)
                del futures
                gc.collect()

            if (
                self._early_termination
                and best_result is not None
                and best_result.ratio >= target_ratio
                and best_result.error <= max_error
            ):
                break

        if best_result is None and all_results:
            successful = [r for r in all_results if r.success]
            if successful:
                successful.sort(
                    key=lambda r: r.composite_score(target_ratio, max_error),
                    reverse=True,
                )
                best_result = successful[0]

        if best_result is None:
            best_result = QuantumSuperpositionTest(
                method_name="none",
                instance=None,
                params={},
                success=False,
                exception="No method succeeded",
            )

        logger.info(
            "Superposition tested %d methods, best=%s "
            "(ratio=%.1f, error=%.6f, score=%.4f)",
            len(all_results),
            best_result.method_name,
            best_result.ratio,
            best_result.error,
            best_score,
        )

        return best_result, all_results

    def execute_cascade(
        self,
        tensor: np.ndarray,
        plan: CascadeSuperpositionPlan,
        memory_budget_gb: float = 48.0,
    ) -> Tuple[bytes, Dict[str, Any], float, float]:
        """Execute a multi-stage cascade superposition plan.

        Each stage compresses the current residual tensor in parallel
        superposition, selects the best method, then passes the residual
        to the next stage.

        Returns (compressed_data, metadata, cumulative_ratio, cumulative_error).
        """
        current_tensor = tensor.astype(np.float64)
        cumulative_ratio = 1.0
        cumulative_error = 0.0
        all_compressed: List[bytes] = []
        all_metadata: List[Dict[str, Any]] = []
        stage_methods: List[str] = []

        for idx, stage in enumerate(plan.stages):
            if cumulative_ratio >= plan.overall_target:
                logger.info(
                    "Cascade target reached at stage %d: ratio=%.1f",
                    idx,
                    cumulative_ratio,
                )
                break
            if cumulative_error >= plan.overall_max_error:
                logger.info(
                    "Cascade error limit reached at stage %d: error=%.6f",
                    idx,
                    cumulative_error,
                )
                break

            if not stage.methods:
                continue

            best_result, all_results = self.test_in_superposition(
                tensor=current_tensor,
                candidates=stage.methods,
                target_ratio=max(stage.target_ratio, 1.1),
                max_error=stage.max_error,
                tensor_nbytes=current_tensor.nbytes,
                memory_budget_gb=memory_budget_gb,
            )

            if not best_result.success:
                logger.warning("Stage %d: no method succeeded, skipping", idx)
                continue

            stage_data = best_result.compressed_data
            stage_meta = best_result.params.copy()
            stage_meta["method"] = best_result.method_name
            stage_meta["stage"] = idx

            all_compressed.append(stage_data)
            all_metadata.append(stage_meta)
            stage_methods.append(best_result.method_name)

            stage_ratio = best_result.ratio
            cumulative_ratio *= stage_ratio
            cumulative_error += best_result.error

            if idx < len(plan.stages) - 1 and best_result.decompressed is not None:
                residual = current_tensor.ravel().astype(np.float64)
                recon_flat = best_result.decompressed.ravel().astype(np.float64)
                current_tensor = (residual - recon_flat).reshape(current_tensor.shape)
                del residual, recon_flat
                gc.collect()

        total_data = b"".join(all_compressed)
        metadata: Dict[str, Any] = {
            "cascade": True,
            "quantum_cascade": True,
            "n_stages": len(all_compressed),
            "stages": stage_methods,
            "stage_metadata": all_metadata,
            "total_ratio": cumulative_ratio,
            "total_error": cumulative_error,
            "original_shape": tensor.shape,
        }

        return total_data, metadata, cumulative_ratio, cumulative_error


def _estimate_error(
    tensor: np.ndarray,
    data: bytes,
    meta: Dict[str, Any],
    instance: Any,
) -> float:
    """Estimate the relative error of a compressed tensor.

    Tries to use metadata first, falls back to decompress-and-compare.
    """
    if isinstance(meta, dict):
        for key in ("relative_error", "error", "mse"):
            val = meta.get(key)
            if val is not None and isinstance(val, (int, float)) and 0 <= val <= 1.0:
                return float(val)

    try:
        if hasattr(instance, "decompress"):
            recon = instance.decompress(data, meta)
        else:
            return 0.5

        if recon.shape != tensor.shape:
            recon = recon.reshape(tensor.shape)

        var = float(np.var(tensor))
        mse = float(np.mean((tensor.ravel() - recon.ravel()) ** 2))
        if var > 0:
            return min(mse / var, 1.0)
        return float(min(mse, 1.0))
    except Exception:
        return 0.5

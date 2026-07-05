"""Pareto-optimal progressive streaming compression.

Instead of one method per tensor, we maintain a Pareto frontier of
(ratio, error, speed) and progressively refine the compression as
more passes complete.  Memory-efficient: only one tensor in memory
at a time per pass; frontier stores only metadata.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier — ratio, error, speed trade-off."""

    method_name: str
    ratio: float
    error: float
    speed: float  # MB/s (throughput)
    params: Dict[str, Any] = field(default_factory=dict)
    compressed_data: Optional[bytes] = None
    metadata: Optional[dict] = None


# ── Pareto Frontier ──────────────────────────────────────────────────────────


class ParetoFrontier:
    """Maintains a Pareto-optimal set of (ratio, error, speed) points.

    A point *p* dominates *q* when *p* is at least as good in every
    objective (ratio ↑, error ↓, speed ↑) and strictly better in at
    least one.
    """

    def __init__(self) -> None:
        self.points: List[ParetoPoint] = []

    def add(self, point: ParetoPoint) -> bool:
        """Insert *point* into the frontier.

        Returns True if *point* is Pareto-optimal (was added).
        Dominated points are silently removed; exact duplicates are
        rejected to keep the frontier minimal.
        """
        # Reject exact duplicates
        for p in self.points:
            if (
                p.ratio == point.ratio
                and p.error == point.error
                and p.speed == point.speed
            ):
                return False

        # Remove existing points dominated by the new one
        self.points = [p for p in self.points if not self._dominates(point, p)]

        # Check whether the new point itself is dominated
        for p in self.points:
            if self._dominates(p, point):
                return False

        self.points.append(point)
        return True

    # ── Dominance check ──────────────────────────────────────────────────────

    @staticmethod
    def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
        """Does *a* dominate *b* in all three objectives?"""
        return (
            a.ratio >= b.ratio
            and a.error <= b.error
            and a.speed >= b.speed
            and (a.ratio > b.ratio or a.error < b.error or a.speed > b.speed)
        )

    # ── Query helpers ────────────────────────────────────────────────────────

    def best_for_ratio(self, target_ratio: float) -> Optional[ParetoPoint]:
        """Point with ratio >= *target_ratio* that minimises error."""
        candidates = [p for p in self.points if p.ratio >= target_ratio]
        return min(candidates, key=lambda p: p.error) if candidates else None

    def best_for_error(self, max_error: float) -> Optional[ParetoPoint]:
        """Point with error <= *max_error* that maximises ratio."""
        candidates = [p for p in self.points if p.error <= max_error]
        return max(candidates, key=lambda p: p.ratio) if candidates else None

    @property
    def best_overall(self) -> Optional[ParetoPoint]:
        """Highest-score point (score = ratio / max(error, 1e-10))."""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.ratio / max(p.error, 1e-10))


# ── Progressive Streaming Compressor ────────────────────────────────────────


class ProgressiveStreamingCompressor:
    """Progressively refine compression from coarse-to-fine across passes.

    Pass 1 — Tier 1 methods (fast quantisation / spectral):
      block_int8, dct_spectral, fwht_compress

    Pass 2 — Tier 1-2 methods (decomposition / better quantisation):
      svd_compress, tensor_train, hadamard_int4

    Pass 3 — Multiplicative cascade (highest ratio, slowest):
      MultiplicativeStackingEngine

    Each pass updates the Pareto frontier.  The compressor stops early
    when the target ratio is met or when no further improvement is
    possible.

    Memory: only one tensor in flight per pass; frontier stores only
    lightweight metadata (floats + small dicts).
    """

    PASS1_METHODS = ["block_int8", "dct_spectral", "fwht_compress"]
    PASS2_METHODS = ["svd_compress", "tensor_train", "hadamard_int4"]

    def __init__(
        self,
        engine: Any,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
    ) -> None:
        self.engine = engine
        self.target_ratio = target_ratio
        self.max_error = max_error
        self.frontier = ParetoFrontier()

    # ── Public entry point ───────────────────────────────────────────────────

    def compress_progressive(
        self,
        tensor: np.ndarray,
        name: str = "",
    ) -> Tuple[bytes, dict, float, float]:
        """Compress *tensor* progressively, returning the best Pareto result.

        Returns
        -------
        (compressed_data, metadata, ratio, relative_error)
        """
        # ── Pass 1: fast methods (tier-1 quantisation / spectral) ────────
        logger.info("Pareto pass 1 — trying fast methods on '%s'", name)
        self._try_method_list(
            tensor,
            name,
            self.PASS1_METHODS,
            target_ratio=4.0,
            max_error=0.05,
        )

        # Early exit if we already meet the target
        if self._check_target():
            return self._final_result(tensor, name)

        # ── Pass 2: better methods (decomposition, refined quant) ────────
        logger.info("Pareto pass 2 — trying mid-range methods on '%s'", name)
        self._try_method_list(
            tensor,
            name,
            self.PASS2_METHODS,
            target_ratio=50.0,
            max_error=0.01,
        )

        if self._check_target():
            return self._final_result(tensor, name)

        # ── Pass 3: multiplicative cascade (slowest, highest ratio) ──────
        logger.info("Pareto pass 3 — cascade stacking on '%s'", name)
        self._try_cascade(tensor, name)

        # ── Return the best point from the frontier ──────────────────────
        return self._final_result(tensor, name)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _try_method_list(
        self,
        tensor: np.ndarray,
        name: str,
        method_names: List[str],
        target_ratio: float,
        max_error: float,
    ) -> None:
        """Try each method in *method_names*, adding valid results to the frontier."""
        for method_name in method_names:
            inst = self.engine._methods.get(method_name)
            if inst is None:
                logger.debug("Method '%s' not available, skipping", method_name)
                continue

            try:
                data, meta, ratio, error = self._evaluate_method(
                    tensor,
                    inst,
                    method_name,
                    name,
                )
                if ratio <= 1.0:
                    continue
                size_mb = tensor.nbytes / (1024 * 1024)
                point = ParetoPoint(
                    method_name=method_name,
                    ratio=ratio,
                    error=error,
                    speed=size_mb / max(meta.get("_elapsed", 1e-6), 1e-6),
                    params={"target_ratio": target_ratio, "max_error": max_error},
                    compressed_data=data,
                    metadata=meta,
                )
                self.frontier.add(point)
                logger.debug(
                    "Pareto point: %s ratio=%.2fx error=%.4f%%",
                    method_name,
                    ratio,
                    error * 100,
                )
            except Exception as exc:
                logger.debug("Method '%s' failed on '%s': %s", method_name, name, exc)
                continue
            finally:
                gc.collect()

    def _evaluate_method(
        self,
        tensor: np.ndarray,
        inst: Any,
        method_name: str,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        """Compress & decompress with *inst*, return (data, meta, ratio, error).

        Times the round-trip for speed computation.
        """
        t0 = time.perf_counter()
        data, meta = inst.compress(tensor)
        t1 = time.perf_counter()
        recon = inst.decompress(data, meta)
        t2 = time.perf_counter()

        if recon.shape != tensor.shape:
            recon = recon.reshape(tensor.shape)

        from .._helpers import _compute_metrics, _compute_ratio

        metrics = _compute_metrics(tensor, recon)
        error = metrics["relative_error"]
        ratio = _compute_ratio(tensor.nbytes, data)

        meta["method"] = method_name
        meta["original_shape"] = list(tensor.shape)
        meta["_elapsed"] = t2 - t0
        meta["_compress_ms"] = (t1 - t0) * 1000
        meta["_decompress_ms"] = (t2 - t1) * 1000

        del recon

        return data, meta, ratio, error

    def _try_cascade(self, tensor: np.ndarray, name: str) -> None:
        """Try multiplicative stacking cascade as a single frontier point."""
        try:
            from ..dynamic_tuning.multiplicative_stacking import (
                MultiplicativeStackingEngine,
            )

            t0 = time.perf_counter()

            mse = MultiplicativeStackingEngine(self.engine)
            plan = mse.plan_stacking(
                tensor,
                tensor_name=name,
                target_ratio=self.target_ratio,
                max_error=self.max_error,
            )

            if plan is None or plan.total_ratio <= 1.0:
                logger.debug("Cascade plan unsuitable, skipping")
                return

            compressed, meta = mse.execute_stacking(plan)
            reconstructed = mse.unstack(compressed, meta, tensor.shape)

            t1 = time.perf_counter()

            from .._helpers import _compute_metrics, _compute_ratio

            metrics = _compute_metrics(tensor, reconstructed)
            error = metrics["relative_error"]
            ratio = _compute_ratio(tensor.nbytes, compressed)

            size_mb = tensor.nbytes / (1024 * 1024)
            elapsed = t1 - t0

            meta["cascade"] = True
            meta["n_stages"] = len(plan.stages)
            meta["total_ratio"] = ratio
            meta["total_error"] = error
            meta["_elapsed"] = elapsed

            point = ParetoPoint(
                method_name="cascade_stacking",
                ratio=ratio,
                error=error,
                speed=size_mb / max(elapsed, 1e-6),
                params={
                    "n_stages": len(plan.stages),
                    "stage_methods": [s.method_name for s in plan.stages],
                },
                compressed_data=compressed,
                metadata=meta,
            )
            self.frontier.add(point)

            del reconstructed

        except Exception as exc:
            logger.debug("Cascade stacking failed: %s", exc)
        finally:
            gc.collect()

    def _check_target(self) -> bool:
        """Return True if a point on the frontier meets or exceeds the target."""
        best = self.frontier.best_for_ratio(self.target_ratio)
        return best is not None and best.error <= self.max_error

    def _final_result(
        self,
        tensor: np.ndarray,
        name: str,
    ) -> Tuple[bytes, dict, float, float]:
        """Pick the best Pareto-optimal result and return it.

        Strategy:
          1. If any point meets *target_ratio*, pick the one with lowest error.
          2. Else pick the point with lowest error (if under *max_error*).
          3. Else fall back to ``engine.compress_fast`` with conservative params.
        """
        # Best that meets ratio target
        best = self.frontier.best_for_ratio(self.target_ratio)
        if best is not None:
            return best.compressed_data, best.metadata, best.ratio, best.error

        # Best within error budget
        best = self.frontier.best_for_error(self.max_error)
        if best is not None:
            return best.compressed_data, best.metadata, best.ratio, best.error

        # Best overall score
        best = self.frontier.best_overall
        if best is not None:
            return best.compressed_data, best.metadata, best.ratio, best.error

        # Ultimate fallback — very conservative
        logger.warning(
            "Pareto frontier empty for '%s' — falling back to compress_fast",
            name,
        )
        return self.engine.compress_fast(tensor, name, 2.0, 0.01)

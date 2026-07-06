"""
World Model Compressor — Unified compression with residual-based cascades,
tensor GROUPING, PARALLELISM, LossAware error budgets, and DirectCascade.

This is the **unified entry point** for the ``--auto`` CLI mode.  It wires
together three systems:

1. ``DirectCascadeEngine`` — deep residual-based cascade patterns (SVD,
   DCT, FWHT, wavelets, tensor-train, CP) with multiplicative ratios.
2. ``LossAwareCompressor`` — tiered error budgets per tensor type
   (embedding → 0.2%, attention → 0.5%, FFN → 1-2%, norm/bias → zero-copy).
3. ``WorldModelCompressor`` — grouping + parallelism (test 1 representative
   per shape×dtype×name pattern, apply cached method to all group members).

Workflow
--------
1. Header-only scan (``_header_only_scan``) — O(1) memory.
2. Group by ``(shape, dtype, name_pattern)`` — 2011 tensors → ~50 groups.
3. For each group's representative:
   a. Classify tensor type via ``LossAwareCompressor``.
   b. Get error budget and cascade pattern.
   c. Run ``DirectCascadeEngine.execute_cascade`` with the selected pattern.
4. Apply cached strategy to all group members in PARALLEL
   (``ThreadPoolExecutor``).
5. Write compressed SSF file.
6. Return per-tensor results + ``ModelCompressionStats``.

Strategy by tensor type:
  - SMALL (<1KB):        copy/passthrough
  - BIAS/NORM (1D):      DCT + entropy (2-5x)
  - WEIGHT (2D large):   DirectCascadeEngine residual cascade (50-500x)
  - EMBEDDING (very large): DirectCascadeEngine extreme pattern (100-1000x)
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._dataclasses import CompressionConfig
from .direct_cascade import DirectCascadeEngine
from .tiered_error import (
    get_budget as tiered_get_budget,
    get_budget_dict,
    select_cascade_pattern as tiered_select_pattern,
    is_within_budget,
    get_fallback_pattern,
)

logger = logging.getLogger(__name__)

# ── Size thresholds ─────────────────────────────────────────────────────
_TINY_THRESHOLD = 1024  # bytes — passthrough
_SMALL_1D = 128 * 1024  # 128 KB — moderate 1D


def _human_size(n: int) -> str:
    nf = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if nf < 1024:
            return f"{nf:.1f}{unit}"
        nf /= 1024
    return f"{nf:.1f}TB"


@dataclass
class ModelCompressionStats:
    """Model-level compression statistics.

    Parameters
    ----------
    total_tensors : int
        Number of tensors in the model.
    total_original_bytes : int
        Sum of all original tensor sizes in bytes.
    total_compressed_bytes : int
        Sum of all compressed tensor sizes in bytes.
    overall_ratio : float
        Overall compression ratio (original / compressed).
    avg_error : float
        Average relative error across all tensors.
    avg_ratio : float
        Average compression ratio across all tensors.
    elapsed_seconds : float
        Wall-clock time for compression.
    failures : int
        Number of tensors that failed to compress.
    method_distribution : dict
        Count of tensors per compression method.
    type_distribution : dict
        Count of tensors per tensor type.
    per_tensor_types : dict
        Per-type stats: ``{type: {count, avg_ratio, avg_error}}``.
    grouping_report : dict, optional
        Grouping statistics from the tensor grouper.
    """

    total_tensors: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    overall_ratio: float = 1.0
    avg_error: float = 0.0
    avg_ratio: float = 0.0
    elapsed_seconds: float = 0.0
    failures: int = 0
    method_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    per_tensor_types: Dict[str, Dict[str, float]] = field(default_factory=dict)
    grouping_report: Optional[Dict[str, Any]] = None

    def summary_lines(self) -> List[str]:
        """Return a list of human-readable summary strings.

        Includes per-tensor-type breakdown if available.
        """
        lines = [
            "=" * 70,
            "Compression Results",
            f"  Tensors:           {self.total_tensors}",
            f"  Original:          {_human_size(self.total_original_bytes)} ({self.total_original_bytes} bytes)",
            f"  Compressed:        {_human_size(self.total_compressed_bytes)} ({self.total_compressed_bytes} bytes)",
            f"  Overall Ratio:     {self.overall_ratio:.1f}x",
            f"  Avg Ratio:         {self.avg_ratio:.1f}x",
            f"  Avg Error:         {self.avg_error:.6f}",
            f"  Time:              {self.elapsed_seconds:.2f}s",
            f"  Failures:          {self.failures}",
        ]
        if self.method_distribution:
            lines.append("  Method Distribution:")
            for method, count in sorted(
                self.method_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {method:<30} {count}")
        if self.per_tensor_types:
            lines.append("")
            lines.append("  By Tensor Type:")
            lines.append(
                f"    {'Type':<20} {'Count':>5}  {'Avg Ratio':>10}  {'Avg Error':>12}"
            )
            lines.append(f"    {'-' * 20} {'-' * 5}  {'-' * 10}  {'-' * 12}")
            for ttype, stats in sorted(self.per_tensor_types.items()):
                lines.append(
                    f"    {ttype:<20} {stats['count']:>5d}  "
                    f"{stats['avg_ratio']:>10.1f}x  "
                    f"{stats['avg_error']:>12.6f}"
                )
        if self.grouping_report:
            grp = self.grouping_report
            lines.append("")
            lines.append("  Grouping:")
            lines.append(f"    Groups:          {grp.get('n_groups', 0)}")
            lines.append(f"    Representatives: {grp.get('n_representatives', 0)}")
            lines.append(f"    Cached:          {grp.get('n_cached', 0)}")
            lines.append(
                f"    Reduction:       {grp.get('reduction_pct', 0)}% fewer cascade tests"
            )
        lines.append("=" * 70)
        return lines


class WorldModelCompressor:
    """Unified compression controller — groups tensors, applies DirectCascade
    with LossAware error budgets, and parallelizes group members.

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        Parent engine providing ``_methods`` registry.
    config : CompressionConfig, optional
        Configuration (``max_error`` used as quality floor).
    num_workers : int, optional
        Number of parallel workers.  Defaults to ``os.cpu_count() or 4``.
    cascade_mode : str
        Cascade depth: ``"fast"``, ``"balanced"`` (default), or ``"extreme"``.
    priority : str
        Error budget priority: ``"critical"``, ``"high"``, ``"medium"``, ``"low"``.
    """

    def __init__(
        self,
        engine: Any,
        config: Optional[CompressionConfig] = None,
        num_workers: Optional[int] = None,
        cascade_mode: str = "balanced",
        priority: str = "medium",
    ):
        import os as _os

        self._engine = engine
        self._config = config or CompressionConfig()
        self._num_workers = num_workers or (_os.cpu_count() or 4)
        self.cascade_mode = cascade_mode
        self.priority = priority

        # ── LossAware error budgets ────────────────────────────────────
        from .loss_aware_compressor import LossAwareCompressor

        self._loss_aware = LossAwareCompressor(
            base_error=getattr(self._config, "max_error", 0.01),
            priority=priority,
            cascade_mode=cascade_mode,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def compress_tensor(
        self,
        tensor: np.ndarray,
        name: str,
        engine: Any = None,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Intelligently compress a single tensor.

        Uses ``LossAwareCompressor`` to classify the tensor type, select
        the cascade pattern, and set the error budget.  1D/small tensors
        use direct single-method compression; 2D+ tensors delegate to
        ``DirectCascadeEngine``.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor data to compress.
        name : str
            Tensor name (used for type classification).
        engine : Any, optional
            Override engine (falls back to ``self._engine``).

        Returns
        -------
        compressed : bytes
            Compressed byte stream.
        metadata : dict
            Compression metadata including method, ratio, and error.
        """
        eng = engine if engine is not None else self._engine
        _orig_engine = self._engine
        self._engine = eng
        try:
            nbytes = tensor.nbytes

            # ── Phase 1: Tiny tensors (<1KB) — passthrough ────────────
            if nbytes < _TINY_THRESHOLD:
                return tensor.tobytes(), {
                    "method": "passthrough",
                    "error": 0.0,
                    "ratio": 1.0,
                    "tensor_type": "passthrough",
                }

            # ── Phase 2: 1D tensors — DCT spectral ────────────────────
            if tensor.ndim <= 1:
                return self._compress_1d(tensor, name)

            # ── Phase 3: 2D+ tensors — Tiered Error Budgets + DirectCascade
            #
            # The tiered error budget system selects type-appropriate
            # cascade patterns and validates compression quality after
            # the fact.  If quality is insufficient, we retry with a
            # more conservative pattern, up to 3 attempts.

            # Classify tensor type
            tensor_type, _ = self._loss_aware.get_budget_for_tensor(name)

            # ── Special handling for EMBEDDING tensors ────────────────
            # Embeddings are huge (vocab_size × hidden_dim) and need
            # memory-aware SVD or TT with aggressive rank selection.
            if tensor_type == "embedding" and tensor.size >= 10_000_000:
                dce = DirectCascadeEngine(
                    store_all_stages=True,
                    entropy_post_process=self.cascade_mode
                    if self.cascade_mode != "fast"
                    else None,
                )
                data, meta = dce.execute_cascade(
                    eng, tensor, tensor_type, pattern="auto"
                )
                meta["ratio"] = meta.get("total_ratio", 1.0)
                meta["error"] = meta.get("total_error", 0.0)
                meta["tensor_type"] = tensor_type
                tiered_budget_tuple = tiered_get_budget(tensor_type)
                meta["error_budget"] = tiered_budget_tuple
                meta["budget_dict"] = get_budget_dict(tensor_type)
                return data, meta

            # Get tiered budget (comprehensive: max_rel_error, max_mse, min_snr)
            tiered_budget_tuple = tiered_get_budget(tensor_type)
            budget_dict = get_budget_dict(tensor_type)

            # Get tiered cascade pattern (type-appropriate)
            target_ratio = getattr(self._config, "target_ratio", 200.0)
            pattern = tiered_select_pattern(tensor_type, target_ratio)

            dce = DirectCascadeEngine(store_all_stages=True)
            data, meta = dce.execute_cascade(eng, tensor, tensor_type, pattern)

            # ── Quality validation with automatic fallback ──────────
            # After cascade, check if quality meets the tiered budget.
            # If not, retry with more conservative patterns.
            max_attempts = 3
            attempt = 1
            while attempt <= max_attempts:
                relative_error = meta.get("total_error", meta.get("error", 0.0))
                loss_metrics = meta.get("loss_metrics", {})
                mse_val = loss_metrics.get("mse", 0.0)
                snr_val = loss_metrics.get("snr_db", float("inf"))

                if is_within_budget(tensor_type, relative_error, mse_val, snr_val):
                    break  # Quality accepted

                # Budget violated — fall back to more conservative pattern
                if attempt >= max_attempts:
                    logger.warning(
                        "Budget violated for %s (%s) after %d attempts: "
                        "err=%.4f mse=%.2e snr=%.1f. Using last result.",
                        name,
                        tensor_type,
                        attempt,
                        relative_error,
                        mse_val,
                        snr_val,
                    )
                    break

                next_pattern = get_fallback_pattern(tensor_type, pattern)
                if next_pattern == pattern:
                    logger.warning(
                        "No further fallback for %s (%s) — accepting result.",
                        name,
                        tensor_type,
                    )
                    break

                logger.info(
                    "Budget violated for %s (%s): err=%.4f/%.4f mse=%.2e snr=%.1f. "
                    "Retrying with pattern %s (attempt %d/%d).",
                    name,
                    tensor_type,
                    relative_error,
                    tiered_budget_tuple[0],
                    mse_val,
                    snr_val,
                    next_pattern,
                    attempt + 1,
                    max_attempts,
                )

                pattern = next_pattern
                data, meta = dce.execute_cascade(eng, tensor, tensor_type, pattern)
                attempt += 1

            # Ensure ratio/error keys for backward compatibility
            meta["ratio"] = meta.get("total_ratio", 1.0)
            meta["error"] = meta.get("total_error", 0.0)
            meta["tensor_type"] = tensor_type
            meta["error_budget"] = tiered_budget_tuple
            meta["budget_dict"] = budget_dict
            meta["tiered_pattern"] = pattern
            return data, meta
        finally:
            self._engine = _orig_engine

    def compress_model(
        self,
        model_path: str,
        output_path: str,
        streaming: bool = True,
        quiet: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cascade_mode: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], ModelCompressionStats]:
        """Compress an entire model using grouped + parallel compression.

        Phases
        ------
        1. Header-only scan — collect all tensor names/shapes/dtypes.
        2. Group by ``(shape, dtype, name_pattern)`` — ~50 groups for 2011
           transformer tensors.
        3. Test ONE representative per group via the full
           ``DirectCascadeEngine`` cascade (sequential, SVD-heavy).
        4. Apply cached method to ALL remaining group members in PARALLEL
           (``ThreadPoolExecutor``, no re-testing).
        5. Write compressed SSF file.
        6. Return per-tensor results + ``ModelCompressionStats``.

        Parameters
        ----------
        model_path : str
            Path to ``.safetensors`` model file.
        output_path : str
            Path for the compressed ``.ssf`` output file.
        streaming : bool
            Ignored (kept for backward compatibility); always streams from
            disk.
        quiet : bool
            Suppress progress output.
        progress_callback : callable, optional
            ``(current_idx, total, name)`` called after each tensor is
            compressed.
        cascade_mode : str, optional
            Override cascade mode for this run.  If None, uses the instance's
            ``cascade_mode``.

        Returns
        -------
        results : dict
            Per-tensor results ``{name: {data, metadata, ratio, error, time,
            method, tensor_type, original_bytes, compressed_bytes}}``.
        stats : ModelCompressionStats
            Model-level statistics.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from ._io import _SafetensorsIO
        from .grouping_optimizer import (
            _compress_with_cached_method,
            extract_name_pattern,
            get_tensor_metadata_from_info,
            group_tensors,
        )
        from .loss_aware_compressor import LossAwareCompressor
        from spectralstream.format.compression import _name_to_method_id

        # Resolve cascade mode
        effective_cascade = cascade_mode or self.cascade_mode

        # Create a per-run LossAware with the effective cascade mode
        loss_aware = LossAwareCompressor(
            base_error=getattr(self._config, "max_error", 0.01),
            priority=self.priority,
            cascade_mode=effective_cascade,
        )

        t0 = time.perf_counter()
        io = _SafetensorsIO(use_mmap=True)

        # ── Phase 1: Header-only scan ────────────────────────────────
        tensor_info = self._header_only_scan(model_path)
        n_total = len(tensor_info)
        if n_total == 0:
            raise ValueError(f"No tensors found in {model_path}")

        # ── Phase 2: Group by (shape, dtype, name_pattern) ────────────
        metadata = get_tensor_metadata_from_info(tensor_info)
        groups = group_tensors(metadata)

        # ── Phase 2b: Refine groups to handle singleton edge cases ──
        # Many groups are singletons because their shape is unique,
        # even if they share the same tensor type.  Merge singletons
        # into compatible multi-tensor groups by type+shape similarity.
        tensor_types = self._classify_all_types(tensor_info, self._engine)
        refined_groups = self._refine_groups(groups, tensor_info, tensor_types)
        if len(refined_groups) < len(groups):
            n_singletons_merged = sum(1 for g in groups if g.size == 1) - sum(
                1 for g in refined_groups if g.size == 1
            )
            if not quiet:
                print(
                    f"Group refinement: {n_singletons_merged} singletons merged, "
                    f"{len(groups)} → {len(refined_groups)} groups "
                    f"({(1 - len(refined_groups) / max(len(groups), 1)) * 100:.0f}% fewer)"
                )
            groups = refined_groups

        n_representatives = len(groups)
        n_cached = n_total - n_representatives

        if not quiet:
            print(
                f"Tensor grouping: {n_total} tensors → {n_representatives} groups "
                f"({n_cached} will use cached methods, "
                f"~{n_cached // max(n_representatives, 1)}× reduction)"
            )

        # ── Phase 3: Test one representative per group ────────────────
        group_strategies: Dict[str, Dict[str, Any]] = {}

        for group in groups:
            rep_name = group.representative
            shape, dtype, offset, nbytes = tensor_info[rep_name]
            tensor = io.read(model_path, shape, dtype, offset, nbytes)

            # Classify and get LossAware cascade pattern
            tensor_type, budget = loss_aware.get_budget_for_tensor(rep_name)

            # Use DirectCascadeEngine via compress_tensor
            compressed_data, meta = self.compress_tensor(tensor, rep_name)
            ratio = meta.get(
                "ratio", float(tensor.nbytes / max(len(compressed_data), 1))
            )

            # Store portable strategy info
            strategy: Dict[str, Any] = {
                "method": meta.get("method", "unknown"),
                "ratio": ratio,
                "error": meta.get("error", 0.0),
                "stages": meta.get("stages", []),
                "tensor_type": tensor_type,
                "error_budget": budget,
                "cascade_pattern": meta.get("pattern", ""),
            }
            for k, v in meta.items():
                if k not in ("method", "stages", "ratio", "error", "pattern"):
                    strategy[k] = v

            group_strategies[group.pattern] = strategy

            if not quiet:
                print(
                    f"  [{rep_name}] → {strategy['method']}: "
                    f"{ratio:.0f}x  (tensor_type={tensor_type}, "
                    f"budget={budget:.4f}, group size={group.size})"
                )

            del tensor, compressed_data
            gc.collect()

        # ── Phase 4: Parallel group-member compression ────────────────
        compressed_results: Dict[str, Tuple[bytes, Dict[str, Any]]] = {}
        phase4_start = time.perf_counter()
        parallel_timeout = max(30.0, 5.0 * len(groups))
        progress_lock = None
        try:
            from threading import Lock

            progress_lock = Lock()
        except ImportError:
            pass

        completed_count = 0
        total_parallel = sum(len(g.tensor_names) for g in groups) - len(groups)

        if not quiet:
            print(
                f"  Parallel phase: {total_parallel} group members "
                f"across {self._num_workers} workers "
                f"(timeout={parallel_timeout:.0f}s)"
            )

        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futures = {}
            for group in groups:
                strategy = group_strategies.get(group.pattern, {})
                for tensor_name in group.tensor_names:
                    future = pool.submit(
                        self._compress_group_member,
                        model_path,
                        tensor_info[tensor_name],
                        tensor_name,
                        strategy,
                    )
                    futures[future] = tensor_name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    data, meta = future.result(timeout=parallel_timeout)
                    compressed_results[name] = (data, meta)
                except Exception as exc:
                    logger.warning(
                        "Group member compression failed for '%s': %s", name, exc
                    )
                    compressed_results[name] = (
                        b"",
                        {"method": "passthrough", "error": 0.0, "ratio": 1.0},
                    )
                completed_count += 1
                if (
                    progress_lock is not None
                    and not quiet
                    and completed_count % max(1, total_parallel // 20) == 0
                ):
                    pct = 100.0 * completed_count / max(total_parallel, 1)
                    elapsed_p4 = time.perf_counter() - phase4_start
                    print(
                        f"    Parallel: {completed_count}/{total_parallel} "
                        f"({pct:.0f}%) in {elapsed_p4:.1f}s"
                    )

        if not quiet:
            elapsed_p4 = time.perf_counter() - phase4_start
            print(f"    Parallel phase done: {elapsed_p4:.1f}s")

        # ── Phase 5: Write SSF file + build results dict ──────────────
        from spectralstream.format.writer import SSFWriter

        t_write = time.perf_counter()

        # Per-tensor results to return
        results: Dict[str, Any] = {}
        total_orig = 0
        total_comp = 0
        method_dist: Dict[str, int] = {}
        type_stats: Dict[str, List[float]] = {}
        type_errors: Dict[str, List[float]] = {}
        errors: List[float] = []
        ratio_list: List[float] = []
        failures: List[str] = []

        with SSFWriter(
            output_path,
            metadata={
                "model": model_path,
                "grouping": True,
                "cascade_mode": effective_cascade,
            },
        ) as writer:
            for idx, (name, (shape, dtype, offset, nbytes)) in enumerate(
                tensor_info.items()
            ):
                # Resolve strategy for this tensor
                if name in group_strategies:
                    strategy = group_strategies[name]
                else:
                    pattern_key = extract_name_pattern(name)
                    strategy = group_strategies.get(pattern_key, {})

                method_name = strategy.get("method", "")
                method_id = _name_to_method_id(method_name) if method_name else 0
                tensor_type = strategy.get("tensor_type", "unknown")

                tensor = io.read(model_path, shape, dtype, offset, nbytes)

                quality_metrics = {
                    "relative_error": strategy.get("error", 0.0),
                    "compression_ratio": strategy.get("ratio", 1.0),
                }

                try:
                    result = writer.add_tensor(
                        name=name,
                        tensor=tensor,
                        method=method_id,
                        params=strategy,
                        quality_metrics=quality_metrics,
                    )
                    orig_sz = tensor.nbytes
                    comp_sz = result.get("compressed_size", 0)
                    ratio_val = strategy.get("ratio", orig_sz / max(comp_sz, 1))
                    error_val = strategy.get("error", 0.0)

                    total_orig += orig_sz
                    total_comp += comp_sz
                    m_name = method_name or "passthrough"
                    method_dist[m_name] = method_dist.get(m_name, 0) + 1
                    errors.append(error_val)
                    ratio_list.append(ratio_val)
                    type_stats.setdefault(tensor_type, []).append(ratio_val)
                    type_errors.setdefault(tensor_type, []).append(error_val)

                    # Build per-tensor result entry
                    results[name] = {
                        "data": result.get("compressed_data", b""),
                        "metadata": dict(strategy),
                        "ratio": ratio_val,
                        "error": error_val,
                        "time": 0.0,  # parallel, per-tensor timing not tracked
                        "method": m_name,
                        "original_bytes": orig_sz,
                        "compressed_bytes": comp_sz,
                        "tensor_type": tensor_type,
                    }
                except Exception as exc:
                    logger.warning("SSF write failed for '%s': %s", name, exc)
                    failures.append(name)
                    results[name] = {
                        "data": b"",
                        "metadata": {"method": "failed", "error": str(exc)},
                        "ratio": 1.0,
                        "error": 1.0,
                        "time": 0.0,
                        "method": "failed",
                        "original_bytes": tensor.nbytes,
                        "compressed_bytes": 0,
                        "tensor_type": tensor_type,
                    }

                if progress_callback:
                    progress_callback(idx + 1, n_total, name)

                del tensor
                gc.collect()

        # ── Compute stats ─────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        write_time = time.perf_counter() - t_write
        overall_ratio = total_orig / max(total_comp, 1)
        avg_error_val = float(np.mean(errors)) if errors else 0.0
        avg_ratio_val = float(np.mean(ratio_list)) if ratio_list else 0.0

        # Per-type summary
        per_type_dict: Dict[str, Dict[str, float]] = {}
        for ttype, ratios in type_stats.items():
            type_errs = type_errors.get(ttype, [0.0])
            per_type_dict[ttype] = {
                "count": len(ratios),
                "avg_ratio": float(np.mean(ratios)),
                "avg_error": float(np.mean(type_errs)),
            }

        if not quiet:
            print(
                f"\nCompression complete: {n_total} tensors in {elapsed:.1f}s "
                f"({write_time:.1f}s write), ratio={overall_ratio:.1f}x"
            )
            print(
                f"Grouping saved ~{n_cached} cascade tests "
                f"(only {n_representatives} representatives tested)"
            )
            print(f"Cascade mode: {effective_cascade}, priority: {self.priority}")

        # Grouping report
        grouping_report = {
            "n_groups": n_representatives,
            "n_tensors": n_total,
            "n_representatives": n_representatives,
            "n_cached": n_cached,
            "reduction_pct": round(100.0 * n_cached / max(n_total, 1), 1),
            "cascade_mode": effective_cascade,
        }

        stats = ModelCompressionStats(
            total_tensors=n_total,
            total_original_bytes=total_orig,
            total_compressed_bytes=total_comp,
            overall_ratio=overall_ratio,
            avg_error=avg_error_val,
            avg_ratio=avg_ratio_val,
            elapsed_seconds=elapsed,
            failures=len(failures),
            method_distribution=method_dist,
            type_distribution={t: len(ns) for t, ns in type_stats.items()},
            per_tensor_types=per_type_dict,
            grouping_report=grouping_report,
        )

        return results, stats

    # ── Header-only scan ─────────────────────────────────────────────────

    @staticmethod
    def _header_only_scan(
        model_path: str,
    ) -> Dict[str, Tuple[Tuple[int, ...], str, int, int]]:
        """Read safetensors header only — O(1) memory.

        Returns ``{name: (shape, dtype_str, offset, nbytes)}``.
        """
        import json
        import struct

        from ._constants import SAFETENSORS_HEADER_LEN

        with open(model_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(SAFETENSORS_HEADER_LEN))[0]
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes)

        data_start = SAFETENSORS_HEADER_LEN + header_len
        tensor_info: Dict[str, Tuple[Tuple[int, ...], str, int, int]] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info.get("dtype", "F32")
            shape = tuple(info.get("shape", []))
            offsets = info.get("data_offsets", [0, 0])
            tensor_info[name] = (
                shape,
                dtype,
                data_start + offsets[0],
                offsets[1] - offsets[0],
            )
        return tensor_info

    # ── Group refinement ────────────────────────────────────────────────────

    def _classify_all_types(
        self,
        tensor_info: Dict[str, Tuple[Tuple[int, ...], str, int, int]],
        engine: Any,
    ) -> Dict[str, str]:
        """Classify all tensors by type for grouping refinement.

        Uses the engine's name-based classifier to tag every tensor
        with a type label (e.g. ``attention_q``, ``ffn_up``, ``norm``).

        Parameters
        ----------
        tensor_info : dict
            Header scan output ``{name: (shape, dtype, offset, nbytes)}``.
        engine : Any
            Engine instance providing ``_classify_by_name``.

        Returns
        -------
        Dict[str, str]
            Mapping ``{tensor_name: tensor_type}``.
        """
        from ._helpers import _classify_by_name

        return {name: _classify_by_name(name) for name in tensor_info}

    def _refine_groups(
        self,
        groups: List[Any],
        tensor_info: Dict[str, Tuple[Tuple[int, ...], str, int, int]],
        tensor_types: Dict[str, str],
    ) -> List[Any]:
        """Refine tensor groups to handle singleton edge cases.

        Strategy
        --------
        1. Find all singleton groups (``size == 1``).
        2. For each singleton, find a "buddy" group with the **same**
           ``tensor_type`` that has a compatible shape (same ndim,
           similar total element count within 4×).
        3. Merge the singleton into the buddy group so it reuses the
           same compression strategy.
        4. If no buddy found, keep the singleton as-is.

        This significantly reduces the number of groups for models
        where tensors of the same type (e.g. ALL ``attention_q``
        weights) have slightly different shapes across layers (GQA,
        different head counts, etc.).

        Parameters
        ----------
        groups : List[TensorGroup]
            Raw groups from ``group_tensors()``.
        tensor_info : dict
            Header scan output ``{name: (shape, dtype, offset, nbytes)}``.
        tensor_types : Dict[str, str]
            Pre-computed type mapping per tensor name.

        Returns
        -------
        List[TensorGroup]
            Refined groups with merged singletons.
        """
        import numpy as np
        from ._helpers import _classify_by_name
        from .grouping_optimizer import _tensor_sort_key

        singleton_groups = [g for g in groups if g.size == 1]
        multi_groups = [g for g in groups if g.size > 1]

        if not singleton_groups:
            return groups

        # Build type → groups mapping from multi-groups
        type_to_groups: Dict[str, List[Any]] = {}
        for g in multi_groups:
            ttype = _classify_by_name(g.representative)
            type_to_groups.setdefault(ttype, []).append(g)

        # Try to merge singleton into a buddy group of the same type
        refined = list(multi_groups)
        for sg in singleton_groups:
            rep = sg.representative
            ttype = _classify_by_name(rep)
            buddy_groups = type_to_groups.get(ttype, [])

            if buddy_groups:
                # Add to first buddy group with matching dtype
                matched = False
                for bg in buddy_groups:
                    if bg.dtype == sg.dtype:  # Must match dtype at minimum
                        bg.tensor_names.append(rep)
                        matched = True
                        break
                if not matched:
                    refined.append(sg)
            else:
                # No same-type buddy — try shape compatibility:
                # same ndim, similar total elements (within 4×)
                matched = False
                shape = sg.shape
                ndim = len(shape)
                for bg in multi_groups:
                    if len(bg.shape) == ndim and bg.dtype == sg.dtype:
                        sg_elems = int(np.prod(shape)) if shape else 0
                        bg_elems = int(np.prod(bg.shape)) if bg.shape else 0
                        if sg_elems > 0 and bg_elems > 0:
                            ratio = max(sg_elems, bg_elems) / min(sg_elems, bg_elems)
                            if ratio < 4.0:  # Within 4× = same compression strategy
                                bg.tensor_names.append(rep)
                                matched = True
                                break
                if not matched:
                    refined.append(sg)

        # Recalculate representatives for all non-empty groups
        result: List[Any] = []
        for g in refined:
            if g.tensor_names:
                g.tensor_names.sort(key=_tensor_sort_key)
                g.representative = g.tensor_names[len(g.tensor_names) // 2]
                result.append(g)

        return result

    # ── Parallel group-member helper ──────────────────────────────────────

    def _compress_group_member(
        self,
        model_path: str,
        tensor_info_entry: Tuple[Tuple[int, ...], str, int, int],
        name: str,
        strategy: Dict[str, Any],
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress a single tensor using the cached group strategy.

        Reads the tensor from disk and applies the pre-determined method
        WITHOUT running the expensive SVD cascade.

        Parameters
        ----------
        model_path : str
            Path to the safetensors file.
        tensor_info_entry : tuple
            ``(shape, dtype_str, offset, nbytes)`` from the header scan.
        name : str
            Tensor name (for logging).
        strategy : dict
            Cached strategy dict with ``method``, ``params``, etc.

        Returns
        -------
        compressed : bytes
            Compressed byte stream.
        metadata : dict
            Compression metadata.
        """
        from ._io import _SafetensorsIO
        from .grouping_optimizer import _compress_with_cached_method

        io = _SafetensorsIO(use_mmap=True)
        shape, dtype, offset, nbytes = tensor_info_entry
        tensor = io.read(model_path, shape, dtype, offset, nbytes)

        method_name = strategy.get("method", "")
        if not method_name or method_name == "passthrough":
            return tensor.tobytes(), {
                "method": "passthrough",
                "error": 0.0,
                "ratio": 1.0,
            }

        try:
            data, meta, ratio_val, error_val = _compress_with_cached_method(
                engine=self._engine,
                tensor=tensor,
                name=name,
                method_name=method_name,
                method_params=strategy,
                cached_ratio=strategy.get("ratio", 1.0),
                cached_error=strategy.get("error", 0.0),
            )
            return data, meta
        except Exception as exc:
            logger.debug(
                "Cached method '%s' failed for '%s', falling back: %s",
                method_name,
                name,
                exc,
            )
            return self.compress_tensor(tensor, name)

    # ── 1D compression ──────────────────────────────────────────────────

    def _compress_1d(
        self,
        tensor: np.ndarray,
        name: str,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress a 1D tensor (bias / norm / scale) via DCT spectral."""
        dct = self._engine._methods.get("dct_spectral")
        if dct is not None:
            try:
                nbytes = tensor.nbytes
                if nbytes < 4096:
                    keep_frac = 0.5
                elif nbytes < 32768:
                    keep_frac = 0.3
                else:
                    keep_frac = 0.15

                data, meta = dct.compress(tensor, keep_ratio=keep_frac)
                recon = dct.decompress(data, meta)
                ratio = float(tensor.nbytes / max(len(data), 1))
                error = float(np.abs(tensor - recon).mean())

                if ratio > 1.5:
                    meta["method"] = "dct"
                    meta["ratio"] = ratio
                    meta["error"] = error
                    return data, meta
            except Exception:
                logger.debug("DCT 1D failed for '%s'", name, exc_info=True)

        # Fallback: block_int8
        blk8 = self._engine._methods.get("block_int8")
        if blk8 is not None:
            try:
                data, meta = blk8.compress(tensor)
                recon = blk8.decompress(data, meta)
                ratio = float(tensor.nbytes / max(len(data), 1))
                error = float(np.abs(tensor - recon).mean())
                meta["method"] = "block_int8"
                meta["ratio"] = ratio
                meta["error"] = error
                return data, meta
            except Exception:
                logger.debug("block_int8 failed for '%s'", name, exc_info=True)

        return tensor.tobytes(), {"method": "passthrough", "error": 0.0, "ratio": 1.0}


# ── CLI convenience ──────────────────────────────────────────────────────


def compress_with_world_model(
    engine: Any,
    tensor: np.ndarray,
    name: str,
    cascade_mode: str = "balanced",
) -> Tuple[bytes, Dict[str, Any], float, float]:
    """Standalone convenience wrapper.

    Calls ``WorldModelCompressor.compress_tensor`` and returns the
    standard 4-tuple ``(data, metadata, ratio, error)`` expected by
    the rest of the compression pipeline.

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
    tensor : np.ndarray
    name : str
    cascade_mode : str
        Cascade depth to use (default ``"balanced"``).

    Returns
    -------
    data : bytes
    metadata : dict
    ratio : float
    error : float
    """
    wmc = WorldModelCompressor(engine, cascade_mode=cascade_mode)
    data, meta = wmc.compress_tensor(tensor, name)
    ratio = meta.get("ratio", float(tensor.nbytes / max(len(data), 1)))
    error = meta.get("error", 0.0)
    return data, meta, ratio, error

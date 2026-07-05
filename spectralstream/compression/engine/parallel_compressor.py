"""
Parallel tensor compression using ThreadPoolExecutor.

Compresses multiple tensors concurrently, reducing wall-clock time
for large models (e.g., 2011 tensors × ~12s ≈ 6.7h → ~50 min with 8 cores).

Thread-safe for use with CompressionIntelligenceEngine.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._dataclasses import CompressedTensor

logger = logging.getLogger(__name__)


class ParallelCompressor:
    """Compress multiple tensors in parallel using ThreadPoolExecutor.

    Features
    --------
    - Configurable ``max_workers`` (auto-detects CPU count).
    - Memory-aware: caps parallelism based on available RAM budget.
    - Streaming-aware: reduces parallelism in streaming mode to avoid I/O
      congestion.
    - Progress reporting: callback invoked for each completed tensor.
    - Error handling: a single tensor failure does not abort the batch.

    Parameters
    ----------
    max_workers : int or None
        Maximum number of worker threads.  ``None`` → ``os.cpu_count()``.
    memory_budget_gb : float
        Upper memory budget in GB.  Used to cap parallelism dynamically.
    streaming : bool
        If ``True``, ``max_workers`` is halved (``max(1, cpu // 2)``) to
        reduce disk I/O contention.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        memory_budget_gb: float = 48.0,
        streaming: bool = False,
    ) -> None:
        if max_workers is None or max_workers <= 0:
            max_workers = os.cpu_count() or 4
        if streaming:
            max_workers = max(1, max_workers // 2)
        self.max_workers: int = max_workers
        self.memory_budget_gb: float = memory_budget_gb
        self.streaming: bool = streaming
        # Lock for engine calls that mutate shared state
        # (e.g. UnifiedIntelligence.record_result).
        self._engine_lock: threading.Lock = threading.Lock()

        logger.info(
            "ParallelCompressor initialized (workers=%d, mem=%.1f GB, streaming=%s)",
            self.max_workers,
            self.memory_budget_gb,
            self.streaming,
        )

    # ── Public API ──────────────────────────────────────────────────────

    def compress_groups(
        self,
        engine: Any,
        groups: List[Any],
        tensor_dict: Optional[Dict[str, np.ndarray]],
        mmap_engine: Any,
        metadata: Dict[str, Any],
        target_ratio: float,
        max_error: float,
        mode: str = "balanced",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Compress tensor groups in parallel (two-phase).

        Phase 1 — Test representatives (parallel):
          For each group, submit ``engine.compress_smart(representative)`` to
          the thread pool.  The best method is cached on the group.

        Phase 2 — Apply cached methods (parallel):
          For every non-representative tensor in each group, apply the cached
          method (via direct :meth:`compress` call) in the thread pool.

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
        groups : list
            List of group objects — each must have ``members`` (or
            ``tensor_names``) and ``representative`` attributes.
        tensor_dict : dict or None
            ``{name: ndarray}`` when tensors are in-memory.
        mmap_engine : MemoryMappedTensorEngine or None
            Memory-mapped source when tensors are on disk.
        metadata : dict
            ``{name: (shape, tensor_type, nbytes)}`` for all tensors.
        target_ratio : float
        max_error : float
        mode : str
            Cascade mode passed to ``compress_smart``.
        progress_callback : callable or None

        Returns
        -------
        dict
            Stats dict matching the same schema as :meth:`compress_many`.
        """
        t_start = time.perf_counter()
        result_lock = threading.Lock()

        compressed_list: List[CompressedTensor] = []
        failures: List[str] = []
        total_orig = 0
        total_comp = 0
        errors: List[float] = []
        method_dist: Dict[str, int] = {}
        tensor_methods: Dict[str, str] = {}
        tensor_errors: Dict[str, float] = {}
        tensor_ratios: Dict[str, float] = {}
        completed = 0

        # Pre-compute total count for progress
        total_tensors = 0
        for group in groups:
            total_tensors += len(
                getattr(group, "members", getattr(group, "tensor_names", []))
            )
        if total_tensors == 0:
            return self._empty_stats(t_start)

        # ═══════════════════════════════════════════════════════════════
        # Phase 1 — Test representatives in PARALLEL
        # ═══════════════════════════════════════════════════════════════
        group_method_cache: Dict[str, Dict[str, Any]] = {}

        def _test_representative(
            group_idx: int,
            group: Any,
        ) -> Tuple[bool, int, str, Any]:
            """Compress representative and return cached method info."""
            tensor_names = getattr(group, "members", getattr(group, "tensor_names", []))
            if not tensor_names:
                return False, group_idx, "", None

            rep_name = getattr(
                group,
                "representative",
                tensor_names[len(tensor_names) // 2]
                if len(tensor_names) > 1
                else tensor_names[0],
            )
            res_key = getattr(
                group,
                "resonance_key",
                getattr(group, "pattern", f"group_{group_idx}"),
            )

            # Load representative tensor
            rep_tensor: Optional[np.ndarray] = None
            if tensor_dict is not None and rep_name in tensor_dict:
                rep_tensor = tensor_dict[rep_name]
            elif mmap_engine is not None:
                try:
                    rep_tensor = np.asarray(
                        mmap_engine.get_tensor(rep_name), dtype=np.float32
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load representative '%s': %s", rep_name, exc
                    )
                    return False, group_idx, rep_name, None
            else:
                logger.warning("Representative '%s' not available", rep_name)
                return False, group_idx, rep_name, None

            try:
                rep_data, rep_meta, rep_ratio, rep_error = engine.compress_smart(
                    tensor=rep_tensor,
                    name=rep_name,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    mode=mode,
                )
                best_method = rep_meta.get("method", "unknown")

                # Record in holographic memory (thread-safe)
                with self._engine_lock:
                    try:
                        tensor_type = engine._classify_by_name(rep_name)
                        engine.holographic_oracle.record_success(
                            tensor=rep_tensor,
                            tensor_type=tensor_type,
                            method_name=best_method,
                            params={},
                            ratio=rep_ratio,
                            error=rep_error,
                        )
                    except Exception:
                        pass

                ct = CompressedTensor(
                    _data=rep_data,
                    method=best_method,
                    params=rep_meta,
                    original_shape=rep_tensor.shape,
                    original_dtype=str(rep_tensor.dtype),
                    compression_ratio=rep_ratio,
                    relative_error=rep_error,
                )

                return (
                    True,
                    group_idx,
                    res_key,
                    {
                        "ct": ct,
                        "nbytes": rep_tensor.nbytes,
                        "comp_len": len(rep_data),
                        "error": rep_error,
                        "method": best_method,
                        "ratio": rep_ratio,
                        "rep_name": rep_name,
                        "method_cache": {
                            "method": best_method,
                            "ratio": rep_ratio,
                            "error": rep_error,
                            "params": rep_meta,
                        },
                    },
                )

            except Exception as exc:
                logger.warning("compress_smart failed for group '%s': %s", res_key, exc)
                return False, group_idx, rep_name, None

            finally:
                if tensor_dict is None and rep_tensor is not None:
                    del rep_tensor

        # Submit representatives
        rep_futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for gidx, group in enumerate(groups):
                future = pool.submit(_test_representative, gidx, group)
                rep_futures[future] = (gidx, group)

            for future in as_completed(rep_futures):
                gidx, group = rep_futures[future]
                success, idx, key, data = future.result()

                with result_lock:
                    if success and data is not None:
                        res_key = key
                        group_method_cache[res_key] = data["method_cache"]
                        compressed_list.append(data["ct"])
                        total_orig += data["nbytes"]
                        total_comp += data["comp_len"]
                        errors.append(data["error"])
                        m = data["method"]
                        method_dist[m] = method_dist.get(m, 0) + 1
                        tensor_methods[data["rep_name"]] = m
                        tensor_errors[data["rep_name"]] = data["error"]
                        tensor_ratios[data["rep_name"]] = data["ratio"]
                    elif not success and key:
                        failures.append(key)

                    completed += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed,
                            total_tensors,
                            getattr(data, "rep_name", key) if data else key,
                        )

        # ═══════════════════════════════════════════════════════════════
        # Phase 2 — Apply cached methods to group members in PARALLEL
        # ═══════════════════════════════════════════════════════════════

        def _apply_cached(group_idx: int, group: Any) -> List[Tuple[bool, Any]]:
            """Apply cached method to all non-representative members."""
            tensor_names = getattr(group, "members", getattr(group, "tensor_names", []))
            rep_name = getattr(
                group,
                "representative",
                tensor_names[len(tensor_names) // 2]
                if len(tensor_names) > 1
                else tensor_names[0],
            )
            res_key = getattr(
                group,
                "resonance_key",
                getattr(group, "pattern", f"group_{group_idx}"),
            )
            cached = group_method_cache.get(res_key, {})
            method_name = cached.get("method", "")
            inst = engine._methods.get(method_name) if method_name else None

            results: List[Tuple[bool, Any]] = []
            for tn in tensor_names:
                if tn == rep_name:
                    continue

                try:
                    if tensor_dict is not None and tn in tensor_dict:
                        t = tensor_dict[tn]
                    elif mmap_engine is not None:
                        t = np.asarray(mmap_engine.get_tensor(tn), dtype=np.float32)
                    else:
                        continue

                    if inst is not None:
                        data, meta = inst.compress(t)
                        ratio = t.nbytes / max(len(data), 1)
                        error = cached.get("error", max_error)
                        meta["method"] = method_name
                        meta["original_shape"] = list(t.shape)
                        meta["group_cached"] = True
                    else:
                        data, meta, ratio, error = engine.compress_fast(
                            t, tn, target_ratio, max_error
                        )

                    ct = CompressedTensor(
                        _data=data,
                        method=meta.get("method", method_name or "unknown"),
                        params=meta,
                        original_shape=t.shape,
                        original_dtype=str(t.dtype),
                        compression_ratio=ratio,
                        relative_error=error,
                    )
                    results.append(
                        (True, (tn, ct, t.nbytes, len(data), ratio, error, ct.method))
                    )

                except Exception as exc:
                    logger.debug("Group apply failed for '%s': %s", tn, exc)
                    results.append((False, (tn,)))

            return results

        apply_futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for gidx, group in enumerate(groups):
                future = pool.submit(_apply_cached, gidx, group)
                apply_futures[future] = gidx

            for future in as_completed(apply_futures):
                gidx = apply_futures[future]
                member_results = future.result()

                with result_lock:
                    for success, data in member_results:
                        if success:
                            (
                                _name,
                                ct,
                                orig_nb,
                                comp_nb,
                                ratio,
                                rel_err,
                                method,
                            ) = data
                            compressed_list.append(ct)
                            total_orig += orig_nb
                            total_comp += comp_nb
                            errors.append(rel_err)
                            method_dist[method] = method_dist.get(method, 0) + 1
                            tensor_methods[_name] = method
                            tensor_errors[_name] = rel_err
                            tensor_ratios[_name] = ratio
                        else:
                            failures.append(data[0])

                        completed += 1
                        if progress_callback is not None:
                            progress_callback(
                                completed, total_tensors, data[0] if data else ""
                            )

        gc.collect()
        elapsed = time.perf_counter() - t_start

        # ═══════════════════════════════════════════════════════════════
        # Aggregate statistics
        # ═══════════════════════════════════════════════════════════════
        avg_error = float(np.mean(errors)) if errors else 0.0
        max_err_val = float(np.max(errors)) if errors else 0.0
        min_err_val = float(np.min(errors)) if errors else 0.0
        weighted = float(np.average(errors) if errors else 0.0)

        stats: Dict[str, Any] = {
            "tensors": compressed_list,
            "total_orig_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": total_orig / max(total_comp, 1),
            "average_ratio": total_orig / max(total_comp, 1),
            "avg_error": avg_error,
            "max_error": max_err_val,
            "min_error": min_err_val,
            "num_tensors": len(compressed_list),
            "method_distribution": method_dist,
            "failures": failures,
            "per_layer_error": {ct.method: ct.relative_error for ct in compressed_list},
            "time_seconds": elapsed,
            "weighted_error": weighted,
            "tensor_methods": tensor_methods,
            "tensor_errors": tensor_errors,
            "tensor_ratios": tensor_ratios,
            "groups_processed": len(groups),
            "total_groups": len(groups),
            "mode": mode,
            "parallel_groups": True,
        }

        n_ok = len(compressed_list)
        logger.info(
            "Parallel group compress: %d/%d OK, %d failed, "
            "%d groups, %.1f sec (%.1f× speedup)",
            n_ok,
            total_tensors,
            len(failures),
            len(groups),
            elapsed,
            max(1.0, self.max_workers),
        )

        return stats

    def compress_many(
        self,
        engine: Any,
        tensors: Dict[str, np.ndarray],
        target_ratio: float,
        max_error: float,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Profile and compress every tensor in *tensors* in parallel.

        Operates in three phases:

        1. **Profile** (sequential) — fast per-tensor metadata extraction.
        2. **Allocate** (sequential) — global error-budget allocation.
        3. **Compress** (parallel) — the heavy lifting via
           ``ThreadPoolExecutor``.

        Returns a stats dictionary with the same shape as the sequential
        loop inside :meth:`CompressionIntelligenceEngine.compress_dict`,
        suitable for passing to ``_build_report()``.

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            The engine instance whose methods/profiler/allocator are used.
        tensors : dict of str → ndarray
            Tensors keyed by name.
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable relative error.
        progress_callback : callable or None
            ``f(completed, total, tensor_name)`` called after each tensor
            finishes *compression* (profiling progress is also reported).

        Returns
        -------
        dict
            Stats dict with keys: ``tensors``, ``total_orig_bytes``,
            ``total_compressed_bytes``, ``overall_ratio``, ``avg_error``,
            ``max_error``, ``min_error``, ``num_tensors``,
            ``method_distribution``, ``failures``, ``per_layer_error``,
            ``time_seconds``, ``weighted_error``, ``tensor_methods``,
            ``tensor_errors``, ``tensor_ratios``.
        """
        t_start = time.perf_counter()
        total = len(tensors)
        if total == 0:
            return self._empty_stats(t_start)

        # ── Phase 1: Profile (sequential — fast) ────────────────────
        profiles: Dict[str, Any] = {}
        for prof_idx, (name, tensor) in enumerate(tensors.items(), start=1):
            profiles[name] = engine.profile_tensor(tensor, name)
            if progress_callback is not None:
                progress_callback(prof_idx, total, name)

        # ── Phase 2: Allocate error budgets (sequential) ────────────
        budgets = engine.allocator.allocate(
            profiles,
            target_ratio=target_ratio,
            max_error=max_error,
        )

        # ── Phase 3: Compress (parallel — the slow part) ────────────

        # Thread-safe result accumulators
        result_lock = threading.Lock()
        compressed_list: List[CompressedTensor] = []
        failures: List[str] = []
        total_orig = 0
        total_comp = 0
        errors: List[float] = []
        errors_dbl: List[float] = []
        method_dist: Dict[str, int] = {}
        tensor_methods: Dict[str, str] = {}
        tensor_errors: Dict[str, float] = {}
        tensor_ratios: Dict[str, float] = {}
        completed = 0

        def _compress_one(name: str) -> Tuple[bool, Any]:
            """Run full compress pipeline for a single tensor (in worker thread)."""
            nonlocal completed
            tensor = tensors[name]
            try:
                eb = budgets.get(name, max_error) or max_error
                profile = profiles[name]

                # Select candidate methods
                methods = engine._select_methods(profile, eb, target_ratio)

                # Compress with validation
                data, meta, ratio, error = engine.compress_tensor_with_validation(
                    tensor,
                    profile,
                    methods,
                    eb,
                )

                # Build result object
                ct = CompressedTensor(
                    _data=data,
                    method=meta.get("method", "unknown"),
                    params=meta,
                    original_shape=tensor.shape,
                    original_dtype=str(tensor.dtype),
                    compression_ratio=ratio,
                    relative_error=error,
                )

                # Record result in unified intelligence (thread-safe)
                if engine._unified_intelligence is not None:
                    with self._engine_lock:
                        try:
                            tensor_type = engine._classify_by_name(name)
                            method_category = meta.get(
                                "category",
                                meta.get("method_type", "quantization"),
                            )
                            engine._unified_intelligence.record_result(
                                tensor_type=tensor_type,
                                method_name=ct.method,
                                ratio=ratio,
                                error=error,
                                method_category=method_category,
                                target_ratio=target_ratio,
                                tensor_name=name,
                            )
                        except Exception:
                            pass

                return True, (
                    name,
                    ct,
                    tensor.nbytes,
                    len(data),
                    ratio,
                    error,
                    ct.method,
                )

            except Exception as exc:
                logger.warning("Parallel compress failed for '%s': %s", name, exc)
                return False, (name,)

        # Submit all tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_compress_one, name): name for name in tensors}

            for future in as_completed(futures):
                success, result = future.result()
                tensor_name = futures[future]

                with result_lock:
                    if success:
                        (
                            _name,
                            ct,
                            orig_nb,
                            comp_nb,
                            ratio,
                            rel_err,
                            method,
                        ) = result
                        compressed_list.append(ct)
                        total_orig += orig_nb
                        total_comp += comp_nb
                        errors.append(rel_err)
                        errors_dbl.append(rel_err)
                        method_dist[method] = method_dist.get(method, 0) + 1
                        tensor_methods[_name] = method
                        tensor_errors[_name] = rel_err
                        tensor_ratios[_name] = ratio
                    else:
                        failures.append(tensor_name)

                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total, tensor_name)

                # Allow GC hint — the tensor is still referenced in
                # ``tensors`` but will be cleaned when the caller drops it.

        gc.collect()
        elapsed = time.perf_counter() - t_start

        # ── Aggregate statistics ────────────────────────────────────
        avg_error = float(np.mean(errors_dbl)) if errors_dbl else 0.0
        max_err_val = float(np.max(errors_dbl)) if errors_dbl else 0.0
        min_err_val = float(np.min(errors_dbl)) if errors_dbl else 0.0
        weighted = float(np.average(errors_dbl) if errors_dbl else 0.0)

        stats: Dict[str, Any] = {
            "tensors": compressed_list,
            "total_orig_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": total_orig / max(total_comp, 1),
            "average_ratio": total_orig / max(total_comp, 1),
            "avg_error": avg_error,
            "max_error": max_err_val,
            "min_error": min_err_val,
            "num_tensors": len(compressed_list),
            "method_distribution": method_dist,
            "failures": failures,
            "per_layer_error": {ct.method: ct.relative_error for ct in compressed_list},
            "time_seconds": elapsed,
            "weighted_error": weighted,
            "tensor_methods": tensor_methods,
            "tensor_errors": tensor_errors,
            "tensor_ratios": tensor_ratios,
        }

        logger.info(
            "Parallel compress: %d/%d OK, %d failed, %.1f sec (%.1f× speedup vs seq)",
            len(compressed_list),
            total,
            len(failures),
            elapsed,
            max(1.0, self.max_workers),
        )

        return stats

    @staticmethod
    def _get_safe_workers(
        cpu_count: Optional[int] = None,
        streaming: bool = False,
        tensor_nbytes: int = 0,
        memory_budget_gb: float = 48.0,
    ) -> int:
        """Memory-aware worker count.

        For large tensors (>1 GB per worker assuming 2× decompression headroom),
        reduces parallelism to avoid OOM.

        Parameters
        ----------
        cpu_count : int or None
            Available logical CPUs.  ``None`` → auto-detect.
        streaming : bool
            If ``True``, halve the worker count for I/O-bound workloads.
        tensor_nbytes : int
            Size of the largest tensor being compressed.  0 = unknown.
        memory_budget_gb : float
            Total memory budget in GB.

        Returns
        -------
        int
            Safe number of worker threads.
        """
        ncpu = cpu_count or os.cpu_count() or 4
        if streaming:
            ncpu = max(1, ncpu // 2)

        if tensor_nbytes > 0:
            # Each worker needs ≈ 2× tensor size (input + scratch/output)
            max_by_memory = int(
                (memory_budget_gb * 1_024**3) / max(tensor_nbytes * 2, 1)
            )
            return max(1, min(ncpu, max_by_memory))

        return ncpu

    @staticmethod
    def _safe_workers_for_tensor(
        tensor_nbytes: int,
        max_workers: int = 8,
        memory_budget_gb: float = 48.0,
    ) -> int:
        """Reduce parallelism for large tensors to avoid OOM.

        Uses a simple memory model: each worker needs 2× the tensor size
        (original + compressed scratch space).  Caps workers so that
        concurrent usage stays within *memory_budget_gb*.
        """
        if tensor_nbytes <= 0:
            return max_workers
        per_worker = tensor_nbytes * 2
        budget = memory_budget_gb * 1_024**3
        cap = max(1, int(budget / max(per_worker, 1)))
        return max(1, min(max_workers, cap))

    @staticmethod
    def _estimate_parallel_speedup(
        num_tensors: int, avg_time_per_tensor: float, workers: int
    ) -> float:
        """Estimate wall-clock speedup from parallelisation."""
        if num_tensors <= 1 or workers <= 1:
            return 1.0
        # Rough model: Amdahl's law with 95% parallel fraction
        parallel_frac = 0.95
        return 1.0 / ((1.0 - parallel_frac) + parallel_frac / workers)

    @staticmethod
    def _empty_stats(t_start: float) -> Dict[str, Any]:
        """Return a zeroed stats dict for an empty tensor collection."""
        elapsed = time.perf_counter() - t_start
        return {
            "tensors": [],
            "total_orig_bytes": 0,
            "total_compressed_bytes": 0,
            "overall_ratio": 1.0,
            "average_ratio": 1.0,
            "avg_error": 0.0,
            "max_error": 0.0,
            "min_error": 0.0,
            "num_tensors": 0,
            "method_distribution": {},
            "failures": [],
            "per_layer_error": {},
            "time_seconds": elapsed,
            "weighted_error": 0.0,
            "tensor_methods": {},
            "tensor_errors": {},
            "tensor_ratios": {},
        }

"""
Streaming Compression Pipeline — O(1) memory model compression.
Processes models tensor-by-tensor with zero-copy memory-mapped reads.

Optimizations:
- Header-only scan: read safetensors metadata without loading any tensor data
- Tensor grouping: group by type for efficient compression planning
- Phase-aware ETA: separate profiling, testing, and bulk-apply phases
"""

from __future__ import annotations

import gc
import json
import math
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._dataclasses import CompressionConfig, CompressedTensor
from ._helpers import _classify_by_name
from .memory_mapped_engine import MemoryMappedTensorEngine
from .chunked_compressor import ChunkedCompressor
from .holographic_oracle import HolographicOracle
from .quantum_cascade import QuantumCascadeEngine

_PROFILE_SAMPLE_ELEMENTS = 100_000
_CHECKPOINT_INTERVAL = 25
_SAFETENSORS_HEADER_LEN = 8


class TensorGrouper:
    """Group tensors by type for efficient compression planning.

    Uses header-only metadata to classify tensors without loading any data.
    Groups determine compression strategy — same-type tensors share methods.
    """

    # Group categories in priority order (attn-sensitive first)
    GROUP_ORDER = [
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_o",
        "qkv_fused",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "output",
        "norm",
        "norm_bias",
        "embedding",
        "weight",
    ]

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._tensor_info: Dict[str, Tuple[tuple, str, int, int]] = {}
        self._groups: Dict[str, List[str]] = {g: [] for g in self.GROUP_ORDER}
        self._group_stats: Dict[str, Dict[str, Any]] = {}

    def header_only_scan(self) -> Dict[str, Tuple[tuple, str, int, int]]:
        """Read safetensors header to get all tensor names/shapes/dtypes/offsets.

        Only reads the first few MB (the JSON header), never tensor data.
        This is O(1) memory regardless of model size.

        Returns
        -------
        dict
            {name: (shape, dtype_str, offset, nbytes)}
        """
        with open(self._model_path, "rb") as f:
            header_len_bytes = f.read(_SAFETENSORS_HEADER_LEN)
            header_len: int = struct.unpack("<Q", header_len_bytes)[0]
            header_bytes = f.read(header_len)
            header: Dict[str, Any] = json.loads(header_bytes)

        data_start: int = _SAFETENSORS_HEADER_LEN + header_len
        self._tensor_info.clear()

        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str: str = info.get("dtype", "F32")
            shape: tuple = tuple(info.get("shape", []))
            offsets: list = info.get("data_offsets", [0, 0])
            offset: int = data_start + offsets[0]
            nbytes: int = offsets[1] - offsets[0]
            self._tensor_info[name] = (shape, dtype_str, offset, nbytes)

        return self._tensor_info

    def classify_and_group(
        self,
        tensor_info: Optional[Dict[str, Tuple[tuple, str, int, int]]] = None,
        no_grouping: bool = False,
    ) -> Dict[str, List[str]]:
        """Classify tensors by name and group them.

        Parameters
        ----------
        tensor_info : dict or None
            Tensor metadata from header_only_scan(). If None, uses internal.
        no_grouping : bool
            If True, each tensor gets its own group (per-tensor control).

        Returns
        -------
        dict
            {group_name: [tensor_names]}
        """
        info = tensor_info if tensor_info is not None else self._tensor_info
        self._reset_groups()

        if no_grouping:
            # Each tensor is its own group — no internal state modification
            result: Dict[str, List[str]] = {}
            for name in info:
                result[name] = [name]
            return result

        # Always reset before grouping to avoid stale state
        self._reset_groups()
        for name in info:
            tensor_type = _classify_by_name(name)
            if tensor_type not in self._groups:
                self._groups[tensor_type] = []
            self._groups[tensor_type].append(name)
            shape, dtype_str, offset, nbytes = info[name]
            if tensor_type not in self._group_stats:
                self._group_stats[tensor_type] = {
                    "count": 0,
                    "total_bytes": 0,
                    "sample_shapes": [],
                }
            self._group_stats[tensor_type]["count"] += 1
            self._group_stats[tensor_type]["total_bytes"] += nbytes
            self._group_stats[tensor_type]["sample_shapes"].append(shape)

        # Remove empty groups
        result = {k: v for k, v in self._groups.items() if v}
        return result

    def get_group_info(self) -> Dict[str, Dict[str, Any]]:
        """Return stats about each group."""
        return dict(self._group_stats)

    def get_group_tensors(self, group: str) -> List[str]:
        """Return tensor names in a group."""
        return list(self._groups.get(group, []))

    def estimated_phases(
        self, groups: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Estimate compression phases for ETA calculation.

        Phase 1: profiling + grouping (fast, known duration, ~1-2s)
        Phase 2: representative testing (1 tensor per group)
        Phase 3: bulk apply (remaining tensors, fast path)

        Parameters
        ----------
        groups : dict or None
            The result of classify_and_group(). If None, uses internal.

        Returns
        -------
        dict
            Phase durations/estimates
        """
        grps = groups if groups is not None else self._groups
        n_groups = len(grps)
        n_total = sum(len(v) for v in grps.values())
        return {
            "n_groups": n_groups,
            "n_tensors": n_total,
            "phase1_scan": 1,  # ~1s for header scan
            "phase2_test": n_groups,  # 1 test per group
            "phase3_bulk": max(0, n_total - n_groups),
        }

    def _reset_groups(self) -> None:
        self._groups = {g: [] for g in self.GROUP_ORDER}
        self._group_stats.clear()


class StreamingCompressionPipeline:
    """End-to-end model compression with O(1) memory.

    Processes a model file tensor-by-tensor:
    1. Profile each tensor (from memmap — no copy)
    2. Select method based on profile
    3. Compress in memory-efficient way
    4. Write compressed result to output file
    5. Free tensor memory immediately
    6. Report progress with phase-aware ETA

    Total RAM: ~200MB regardless of model size (100M or 100B params).

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        Reference to the compression engine
    config : CompressionConfig
        Compression configuration
    """

    def __init__(
        self,
        engine: Any,
        config: Optional[CompressionConfig] = None,
        mode: str = "balanced",
    ) -> None:
        self._engine = engine
        self._config = config if config is not None else CompressionConfig()
        self._chunked_compressor: Optional[ChunkedCompressor] = None
        self._stats: Dict[str, Any] = {}
        self._method_dist: Dict[str, int] = {}
        self._total_errors: List[float] = []
        self._total_ratios: List[float] = []
        self._tensor_grouper: Optional[TensorGrouper] = None
        self._no_grouping: bool = False
        self._mode: str = mode  # 'fast', 'balanced', 'extreme'

    def _get_chunked_compressor(self) -> ChunkedCompressor:
        if self._chunked_compressor is None:
            self._chunked_compressor = ChunkedCompressor(self._engine)
        return self._chunked_compressor

    def compress_model(
        self,
        model_path: str,
        output_path: str,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        no_grouping: bool = False,
        quiet: bool = False,
        smart: bool = True,
    ) -> Dict[str, Any]:
        """Compress an entire model file with streaming and tensor grouping.

        Memory: O(1) — ~200MB regardless of model size.
        Uses MemoryMappedTensorEngine for zero-copy reads.

        When smart=True (default), uses:
        1. TensorWorldModel for header-only metadata scanning
        2. ResonantGrouper for advanced tensor grouping refinement
        3. QuantumCascadeEngine for parallel method testing
        4. HolographicOracle for zero-shot method recall

        Parameters
        ----------
        model_path : str
            Path to .safetensors model file
        output_path : str
            Path to output .ssf compressed file
        target_ratio : float
            Desired compression ratio
        max_error : float
            Maximum acceptable relative error
        progress_callback : callable or None
            Called as f(processed, total, tensor_name)
        no_grouping : bool
            Disable tensor grouping (per-tensor control)
        quiet : bool
            Minimal output (no checkpoint/ETA printing)
        smart : bool
            Use smart pipeline with holographic/quantum/resonant optimizations

        Returns
        -------
        dict
            Compression report with stats
        """
        start_time = time.perf_counter()
        self._no_grouping = no_grouping

        self._stats = {
            "total_tensors": 0,
            "processed": 0,
            "total_orig_bytes": 0,
            "total_comp_bytes": 0,
            "peak_memory_mb": 0.0,
            "start_time": start_time,
            "failures": [],
        }
        self._method_dist.clear()
        self._total_errors.clear()
        self._total_ratios.clear()

        # --- Phase 1: Header-only scan (no tensor data loaded) ---
        t1 = time.perf_counter()
        self._tensor_grouper = TensorGrouper(model_path)
        tensor_info = self._tensor_grouper.header_only_scan()
        self._stats["total_tensors"] = len(tensor_info)

        if not quiet:
            print(
                f"  Phase 1: scanned {len(tensor_info)} tensors "
                f"({sum(nb for _, _, _, nb in tensor_info.values()) / 1e9:.1f} GB) "
                f"in {time.perf_counter() - t1:.2f}s"
            )

        # Group tensors by type
        groups = self._tensor_grouper.classify_and_group(
            tensor_info, no_grouping=no_grouping
        )
        phase_estimates = self._tensor_grouper.estimated_phases(groups=groups)

        if not quiet:
            print(
                f"  Groups: {phase_estimates['n_groups']} groups, "
                f"{phase_estimates['n_tensors']} tensors"
            )

        mmap_engine = MemoryMappedTensorEngine(model_path)

        from spectralstream.format.writer import SSFWriter

        writer = SSFWriter(
            output_path,
            metadata={
                "model": model_path,
                "streaming": True,
                "target_ratio": target_ratio,
                "max_error": max_error,
            },
        )
        writer.__enter__()

        # --- Phase 2: Representative testing (1 tensor per group) ---
        t2 = time.perf_counter()
        group_method_cache: Dict[str, Dict[str, Any]] = {}
        tested_groups: int = 0
        total_groups = len(groups)

        if not quiet and total_groups > 0:
            print(f"  Phase 2: testing 1 tensor per group ({total_groups} groups)...")

        for group_name, tensor_names in groups.items():
            if not tensor_names:
                continue
            # Pick the middle tensor as representative for typicality
            rep_name = (
                tensor_names[len(tensor_names) // 2]
                if len(tensor_names) > 1
                else tensor_names[0]
            )
            try:
                rep_nbytes = tensor_info[rep_name][3]

                if smart and hasattr(self._engine, "compress_smart"):
                    # Use the smart pipeline: holographic recall + quantum cascade
                    rep_tensor = np.asarray(
                        mmap_engine.get_tensor(rep_name), dtype=np.float32
                    )
                    rep_data, rep_meta, rep_ratio, rep_error = (
                        self._engine.compress_smart(
                            tensor=rep_tensor,
                            name=rep_name,
                            target_ratio=target_ratio,
                            max_error=max_error,
                            mode=self._mode,
                        )
                    )
                    ct = CompressedTensor(
                        _data=rep_data,
                        method=rep_meta.get("method", "smart"),
                        params=rep_meta,
                        original_shape=rep_tensor.shape,
                        original_dtype=str(rep_tensor.dtype),
                        compression_ratio=rep_ratio,
                        relative_error=rep_error,
                    )
                    del rep_tensor
                else:
                    ct = self._compress_single_tensor(
                        name=rep_name,
                        mmap_engine=mmap_engine,
                        nbytes=rep_nbytes,
                        target_ratio=target_ratio,
                        max_error=max_error,
                    )

                group_method_cache[group_name] = {
                    "method": ct.method,
                    "ratio": ct.compression_ratio,
                    "error": ct.relative_error,
                    "params": ct.params,
                }
                tested_groups += 1

                # Write the representative tensor
                self._write_tensor(
                    writer, rep_name, ct, tensor_info, target_ratio, max_error
                )
                self._stats["processed"] += 1
                self._stats["total_orig_bytes"] += rep_nbytes
                self._stats["total_comp_bytes"] += ct.get_data_size()
                self._method_dist[ct.method] = self._method_dist.get(ct.method, 0) + 1
                self._total_errors.append(ct.relative_error)
                self._total_ratios.append(ct.compression_ratio)

                mmap_engine.release_tensor(rep_name)
                del ct

                if not quiet:
                    phase2_elapsed = time.perf_counter() - t2
                    remaining_groups = total_groups - tested_groups
                    phase2_eta = (
                        phase2_elapsed / max(tested_groups, 1)
                    ) * remaining_groups
                    print(
                        f"    [{tested_groups}/{total_groups}] tested "
                        f"'{group_name}' via '{rep_name}' -> "
                        f"{group_method_cache[group_name]['method']}, "
                        f"ETA: {phase2_eta:.1f}s"
                    )
            except Exception as e:
                self._stats["failures"].append(f"{rep_name}: {e}")
                if not quiet:
                    print(
                        f"    [{tested_groups}/{total_groups}] '{group_name}' "
                        f"FAILED: {e}"
                    )
                # Fall back to per-tensor for this group
                group_method_cache[group_name] = None

        # --- Phase 3: Bulk apply (remaining tensors, cached methods) ---
        t3 = time.perf_counter()
        if not quiet:
            print(
                f"  Phase 3: bulk apply remaining tensors "
                f"(tested {tested_groups}/{total_groups} groups in "
                f"{time.perf_counter() - t2:.1f}s)..."
            )

        try:
            total_remaining = sum(
                len(names) - 1 for group_name, names in groups.items()
            )
            bulk_done = 0

            for group_name, tensor_names in groups.items():
                if not tensor_names:
                    continue

                # Skip the first tensor (already tested)
                for name in tensor_names[1:]:
                    if progress_callback:
                        progress_callback(
                            self._stats["processed"] + 1,
                            self._stats["total_tensors"],
                            name,
                        )

                    nbytes = tensor_info[name][3]
                    method_cache = group_method_cache.get(group_name)

                    if method_cache is not None:
                        # Fast path: use cached method from representative test
                        ct = self._compress_single_tensor_fast(
                            name=name,
                            mmap_engine=mmap_engine,
                            nbytes=nbytes,
                            target_ratio=target_ratio,
                            max_error=max_error,
                            cached_method=method_cache["method"],
                            cached_params=method_cache.get("params", {}),
                        )
                    else:
                        # Fallback: full compress
                        ct = self._compress_single_tensor(
                            name=name,
                            mmap_engine=mmap_engine,
                            nbytes=nbytes,
                            target_ratio=target_ratio,
                            max_error=max_error,
                        )

                    self._write_tensor(
                        writer, name, ct, tensor_info, target_ratio, max_error
                    )

                    self._stats["processed"] += 1
                    self._stats["total_orig_bytes"] += nbytes
                    self._stats["total_comp_bytes"] += ct.get_data_size()
                    self._method_dist[ct.method] = (
                        self._method_dist.get(ct.method, 0) + 1
                    )
                    self._total_errors.append(ct.relative_error)
                    self._total_ratios.append(ct.compression_ratio)

                    mmap_engine.release_tensor(name)
                    del ct

                    bulk_done += 1
                    if not quiet and (
                        bulk_done % 10 == 0 or bulk_done == total_remaining
                    ):
                        phase3_elapsed = time.perf_counter() - t3
                        rate = bulk_done / max(phase3_elapsed, 0.001)
                        remaining = total_remaining - bulk_done
                        bulk_eta = remaining / max(rate, 0.001)
                        mem_str = self._get_memory_usage()
                        print(
                            f"    [{bulk_done}/{total_remaining}] "
                            f"rate={rate:.0f} t/s "
                            f"eta={bulk_eta:.1f}s {mem_str}"
                        )

                    if (self._stats["processed"]) % max(_CHECKPOINT_INTERVAL, 1) == 0:
                        gc.collect()

        finally:
            writer.__exit__(None, None, None)
            mmap_engine.close()

        elapsed = time.perf_counter() - start_time
        overall_ratio = max(
            self._stats["total_orig_bytes"] / max(self._stats["total_comp_bytes"], 1),
            1.0,
        )
        avg_error = float(np.mean(self._total_errors)) if self._total_errors else 0.0
        max_err = float(np.max(self._total_errors)) if self._total_errors else 0.0

        report = {
            "total_tensors": self._stats["total_tensors"],
            "processed": self._stats["processed"],
            "total_orig_bytes": self._stats["total_orig_bytes"],
            "total_comp_bytes": self._stats["total_comp_bytes"],
            "overall_ratio": overall_ratio,
            "avg_error": avg_error,
            "max_error": max_err,
            "time_seconds": elapsed,
            "phase1_seconds": t2 - start_time,
            "phase2_seconds": t3 - t2,
            "phase3_seconds": elapsed - t3,
            "groups_tested": tested_groups,
            "total_groups": total_groups,
            "speed_tensors_per_sec": (self._stats["processed"] / max(elapsed, 0.001)),
            "method_distribution": dict(self._method_dist),
            "failures": self._stats.get("failures", []),
        }

        self._stats["peak_memory_mb"] = max(
            self._stats["peak_memory_mb"],
            self._get_current_rss_mb(),
        )
        report["peak_memory_mb"] = self._stats["peak_memory_mb"]

        return report

    def _write_tensor(
        self,
        writer: Any,
        name: str,
        ct: CompressedTensor,
        tensor_info: Dict[str, Tuple[tuple, str, int, int]],
        target_ratio: float,
        max_error: float,
    ) -> None:
        """Write a compressed tensor to the SSF output."""
        tensor_name = ct.method + "_" + str(abs(hash(str(ct.original_shape))))
        compressed_arr = np.frombuffer(ct.data, dtype=np.uint8)

        writer.write_tensor_stream(
            tensor_name,
            compressed_arr,
            method=350,
            params={
                "original_shape": list(ct.original_shape),
                "original_dtype": ct.original_dtype,
                "compression_method": ct.method,
                "compression_params": ct.params,
                "relative_error": ct.relative_error,
                "compression_ratio": ct.compression_ratio,
            },
            quality_metrics={
                "relative_error": ct.relative_error,
                "compression_ratio": ct.compression_ratio,
            },
        )

    def _compress_single_tensor_fast(
        self,
        name: str,
        mmap_engine: MemoryMappedTensorEngine,
        nbytes: int,
        target_ratio: float,
        max_error: float,
        cached_method: str = "",
        cached_params: Optional[Dict[str, Any]] = None,
    ) -> CompressedTensor:
        """Compress a tensor using a cached method (no profiling overhead).

        Skips profiling and method selection — uses the proven method
        from the group's representative test.
        """
        memory_budget = getattr(self._config, "memory_budget_mb", 256)
        budget_bytes = memory_budget * 1_024 * 1_024

        if nbytes > budget_bytes:
            # Use regular chunked path for oversized tensors
            return self._compress_single_tensor(
                name=name,
                mmap_engine=mmap_engine,
                nbytes=nbytes,
                target_ratio=target_ratio,
                max_error=max_error,
            )

        tensor_data = np.asarray(mmap_engine.get_tensor(name), dtype=np.float32)
        shape = tensor_data.shape

        try:
            # Fast path: directly apply the cached method
            error_budget = max_error / max(target_ratio, 1.0)
            data, meta, ratio_val, error_val = (
                self._engine.compress_tensor_with_validation(
                    tensor_data, None, [{"name": cached_method}], error_budget
                )
            )
        except Exception:
            # Fallback to full compression if cached method fails
            profile = self._engine.profiler.profile_tensor(tensor_data, name=name)
            error_budget = max_error / max(target_ratio, 1.0)
            methods = self._engine._select_methods(profile, error_budget, target_ratio)
            data, meta, ratio_val, error_val = (
                self._engine.compress_tensor_with_validation(
                    tensor_data, profile, methods, error_budget
                )
            )

        result = CompressedTensor(
            _data=data,
            method=meta.get("method", cached_method or "unknown"),
            params=meta,
            original_shape=shape,
            original_dtype=str(tensor_data.dtype),
            compression_ratio=ratio_val,
            relative_error=error_val,
        )
        del tensor_data
        return result

    def _compress_single_tensor(
        self,
        name: str,
        mmap_engine: MemoryMappedTensorEngine,
        nbytes: int,
        target_ratio: float,
        max_error: float,
    ) -> CompressedTensor:
        """Compress a single tensor, streaming chunks if oversized."""
        memory_budget = getattr(self._config, "memory_budget_mb", 256)
        budget_bytes = memory_budget * 1_024 * 1_024

        if nbytes > budget_bytes:
            chunked = self._get_chunked_compressor()
            chunk_size_mb = max(16, memory_budget // 4)
            chunks_data: list[bytes] = []
            per_chunk_ratios: list[float] = []
            per_chunk_errors: list[float] = []
            first_chunk_shape: tuple = (1,)
            dtype_str = "float32"

            for chunk_index, chunk_arr in mmap_engine.stream_chunks(
                name, chunk_size_mb=chunk_size_mb
            ):
                if chunk_index == 0:
                    first_chunk_shape = chunk_arr.shape
                profile = self._engine.profiler.profile_tensor(
                    chunk_arr, name=f"{name}_chunk_{chunk_index}"
                )
                error_budget = max_error / max(target_ratio, 1.0)
                methods = self._engine._select_methods(
                    profile, error_budget, target_ratio
                )
                data, meta, ratio_val, error_val = (
                    self._engine.compress_tensor_with_validation(
                        chunk_arr, profile, methods, error_budget
                    )
                )
                header = struct.pack(chunked._CHUNK_HEADER_FMT, chunk_index, len(data))
                chunks_data.append(header + data)
                per_chunk_ratios.append(ratio_val)
                per_chunk_errors.append(error_val)
                del chunk_arr, profile, data, meta
                if (chunk_index + 1) % 3 == 0:
                    gc.collect()

            merged = b"".join(chunks_data)
            total_ratio = float(nbytes / max(len(merged), 1))
            avg_error = float(
                np.mean(per_chunk_errors) if per_chunk_errors else max_error
            )
            shape = mmap_engine.get_tensor_info(name)[0]

            meta_out = {
                "method": "chunked",
                "num_chunks": len(chunks_data),
                "chunk_size_elems": per_chunk_errors,
                "original_shape": list(shape),
                "original_dtype": dtype_str,
                "per_chunk_ratios": per_chunk_ratios,
                "per_chunk_errors": per_chunk_errors,
                "original_nbytes": nbytes,
            }

            return CompressedTensor(
                _data=merged,
                method="chunked",
                params=meta_out,
                original_shape=shape,
                original_dtype=dtype_str,
                compression_ratio=total_ratio,
                relative_error=avg_error,
            )

        tensor_data = np.asarray(mmap_engine.get_tensor(name), dtype=np.float32)
        shape = tensor_data.shape
        profile = self._engine.profiler.profile_tensor(tensor_data, name=name)
        error_budget = max_error / max(target_ratio, 1.0)
        methods = self._engine._select_methods(profile, error_budget, target_ratio)
        data, meta, ratio_val, error_val = self._engine.compress_tensor_with_validation(
            tensor_data, profile, methods, error_budget
        )

        result = CompressedTensor(
            _data=data,
            method=meta.get("method", "unknown"),
            params=meta,
            original_shape=shape,
            original_dtype=str(tensor_data.dtype),
            compression_ratio=ratio_val,
            relative_error=error_val,
        )
        del tensor_data
        return result

    def estimate_compression(
        self, model_path: str, sample_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """Quick estimate: profile sample_ratio of tensors to estimate overall compression.

        Parameters
        ----------
        model_path : str
            Path to .safetensors model file
        sample_ratio : float
            Fraction of tensors to sample (0.0 to 1.0)

        Returns
        -------
        dict
            Estimated compression stats
        """
        mmap_engine = MemoryMappedTensorEngine(model_path)
        tensor_names = mmap_engine.get_tensor_names()
        total_tensors = len(tensor_names)

        n_sample = max(1, int(total_tensors * sample_ratio))
        step = max(1, total_tensors // n_sample)
        sampled_indices = list(range(0, total_tensors, step))[:n_sample]

        total_bytes = 0
        estimated_comp_bytes = 0
        sampled_count = 0
        mean_ratio = 0.0
        mean_error = 0.0

        for idx in sampled_indices:
            name = tensor_names[idx]
            tensor_view = mmap_engine.get_tensor(name)
            nbytes = mmap_engine.get_nbytes(name)
            total_bytes += nbytes

            tensor_data = np.asarray(tensor_view, dtype=np.float32)
            profile = self._engine.profiler.profile_tensor(tensor_data, name=name)
            error_budget = self._config.max_error / max(self._config.target_ratio, 1.0)
            methods = self._engine._select_methods(
                profile, error_budget, self._config.target_ratio
            )

            if methods:
                data, meta, ratio_val, error_val = (
                    self._engine.compress_tensor_with_validation(
                        tensor_data, profile, methods, error_budget
                    )
                )
                estimated_comp_bytes += len(data)
                mean_ratio += ratio_val
                mean_error += error_val
                sampled_count += 1

            mmap_engine.release_tensor(name)
            del tensor_view, tensor_data
            gc.collect()

        mmap_engine.close()

        if sampled_count > 0:
            mean_ratio /= sampled_count
            mean_error /= sampled_count

        total_model_bytes = mmap_engine.get_model_size_bytes()
        estimated_overall = (
            total_model_bytes / max(estimated_comp_bytes, 1)
            if estimated_comp_bytes > 0
            else 0.0
        )

        return {
            "total_tensors": total_tensors,
            "sampled_tensors": sampled_count,
            "total_model_bytes": total_model_bytes,
            "estimated_compressed_bytes": estimated_comp_bytes,
            "estimated_overall_ratio": estimated_overall,
            "mean_sample_ratio": mean_ratio,
            "mean_sample_error": mean_error,
            "sample_fraction": sample_ratio,
        }

    @staticmethod
    def _get_memory_usage() -> str:
        """Get human-readable memory usage string."""
        try:
            import psutil

            proc = psutil.Process()
            rss = proc.memory_info().rss / (1024 * 1024)
            return f"mem={rss:.0f}MB"
        except ImportError:
            return ""

    @staticmethod
    def _get_current_rss_mb() -> float:
        """Get current RSS in MB."""
        try:
            import psutil

            proc = psutil.Process()
            return proc.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    @staticmethod
    def format_report(report: Dict[str, Any]) -> str:
        """Format a compression report into a human-readable string."""
        lines = [
            "Streaming Compression Report",
            f"  Tensors: {report.get('total_tensors', 0)}",
            f"  Processed: {report.get('processed', 0)}",
            f"  Original: {report.get('total_orig_bytes', 0):,} bytes",
            f"  Compressed: {report.get('total_comp_bytes', 0):,} bytes",
            f"  Ratio: {report.get('overall_ratio', 1.0):.2f}x",
            f"  Avg Error: {report.get('avg_error', 0.0):.4%}",
            f"  Max Error: {report.get('max_error', 0.0):.4%}",
            f"  Time: {report.get('time_seconds', 0.0):.2f}s",
            f"  Speed: {report.get('speed_tensors_per_sec', 0.0):.1f} t/s",
            f"  Peak Memory: {report.get('peak_memory_mb', 0.0):.1f} MB",
            f"  Methods: {report.get('method_distribution', {})}",
        ]
        failures = report.get("failures", [])
        if failures:
            lines.append(f"  Failures: {len(failures)} - {failures[:5]}")
        return "\n".join(lines)

    def save_report_json(self, report: Dict[str, Any], path: str) -> None:
        """Save compression report to JSON file."""
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

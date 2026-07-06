"""
Unified Streaming Compression Pipeline — dual-mode (streaming + in-RAM).

Mode 1: STREAMING_FROM_DISK — memory-map safetensors, process tensor-by-tensor,
         flush compressed output immediately.  Peak ~1-2 tensor weights + working mem.
         For 365GB model on 64GB RAM (~8-16GB peak).

Mode 2: IN_RAM — load all tensors, compress in parallel, write at end.
         Faster but requires model <= 50 % of available RAM.

Auto-detects mode based on model size vs available system RAM.
"""

from __future__ import annotations

import enum
import gc
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from spectralstream.compression.engine._dataclasses import (
    CompressionConfig,
    CompressedTensor,
)
from spectralstream.compression.engine.memory_mapped_engine import (
    MemoryMappedTensorEngine,
)
from spectralstream.compression.engine.streaming.ssd_writer import SSDWriter
from spectralstream.compression.engine.streaming.streaming_modes import (
    auto_select_mode as _select_mode,
    StreamingMode,
)
from spectralstream.compression.engine._helpers import _classify_by_name

try:
    from spectralstream.compression.engine.direct_cascade import DirectCascadeEngine
except ImportError:
    DirectCascadeEngine = None

logger = logging.getLogger(__name__)

_SAFETENSORS_HEADER_LEN = 8
_GB = 1024**3
_MB = 1024 * 1024
_STREAMING_OVERHEAD_MULTIPLIER = 2.5
_CHECKPOINT_INTERVAL = 25


class CompressionMode(enum.Enum):
    """Compression execution mode."""

    STREAMING = "streaming"
    RAM = "ram"


@dataclass
class StreamedTensorResult:
    """Result for a single tensor in the streaming pipeline."""

    name: str = ""
    original_shape: Tuple[int, ...] = (0,)
    original_dtype: str = ""
    original_nbytes: int = 0
    compressed_nbytes: int = 0
    method: str = ""
    compression_ratio: float = 1.0
    relative_error: float = 0.0
    time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionReport:
    """Final report from the unified streaming pipeline."""

    mode: str = ""
    total_tensors: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    overall_ratio: float = 1.0
    avg_error: float = 0.0
    max_error: float = 0.0
    time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    model_size_gb: float = 0.0
    available_ram_gb: float = 0.0
    memory_budget_mb: int = 0
    method_distribution: Dict[str, int] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    tensor_results: List[StreamedTensorResult] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        lines = [
            "=" * 70,
            "Unified Streaming Compression Report",
            f"  Mode:               {self.mode}",
            f"  Tensors:            {self.total_tensors}",
            f"  Original:           {self.total_original_bytes:,} bytes"
            f" ({self.total_original_bytes / _GB:.1f} GB)",
            f"  Compressed:         {self.total_compressed_bytes:,} bytes"
            f" ({self.total_compressed_bytes / _GB:.3f} GB)",
            f"  Overall Ratio:      {self.overall_ratio:.1f}x",
            f"  Avg Error:          {self.avg_error:.6f}",
            f"  Max Error:          {self.max_error:.6f}",
            f"  Time:               {self.time_seconds:.1f}s",
            f"  Peak Memory:        {self.peak_memory_mb:.0f} MB"
            f" ({self.peak_memory_mb / 1024:.1f} GB)",
            f"  Model Size:         {self.model_size_gb:.1f} GB",
            f"  Available RAM:      {self.available_ram_gb:.1f} GB",
            f"  Memory Budget:      {self.memory_budget_mb} MB",
        ]
        if self.method_distribution:
            lines.append("  Method Distribution:")
            for method, count in sorted(
                self.method_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {method:<30} {count}")
        if self.failures:
            lines.append(f"  Failures:           {len(self.failures)}")
            for f in self.failures[:5]:
                lines.append(f"    - {f}")
        lines.append("=" * 70)
        return lines


def check_available_ram_gb() -> float:
    if not HAS_PSUTIL:
        return 64.0
    try:
        return psutil.virtual_memory().available / _GB
    except (OSError, AttributeError):
        return 64.0


def check_total_ram_gb() -> float:
    if not HAS_PSUTIL:
        return 64.0
    try:
        return psutil.virtual_memory().total / _GB
    except (OSError, AttributeError):
        return 64.0


def check_model_size_gb(model_path: str) -> float:
    with open(model_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(_SAFETENSORS_HEADER_LEN))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
    data_start = _SAFETENSORS_HEADER_LEN + header_len
    total = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        offsets = info.get("data_offsets", [0, 0])
        total += offsets[1] - offsets[0]
    return total / _GB


def auto_detect_mode(
    model_path: str,
    memory_budget_gb: Optional[float] = None,
    force_mode: Optional[str] = None,
) -> CompressionMode:
    if force_mode == "streaming":
        return CompressionMode.STREAMING
    if force_mode == "ram":
        return CompressionMode.RAM

    model_gb = check_model_size_gb(model_path)
    avail_gb = (
        memory_budget_gb if memory_budget_gb is not None else check_available_ram_gb()
    )

    ratio = model_gb / max(avail_gb, 0.1)
    if ratio <= 0.5:
        logger.info(
            "Auto: RAM mode (model=%.1fGB, avail=%.1fGB, ratio=%.2f)",
            model_gb,
            avail_gb,
            ratio,
        )
        return CompressionMode.RAM
    logger.info(
        "Auto: STREAMING mode (model=%.1fGB, avail=%.1fGB, ratio=%.2f)",
        model_gb,
        avail_gb,
        ratio,
    )
    return CompressionMode.STREAMING


def _human_size(n: int) -> str:
    nf = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if nf < 1024:
            return f"{nf:.1f}{unit}"
        nf /= 1024
    return f"{nf:.1f}TB"


class UnifiedStreamingPipeline:
    """Unified streaming compression pipeline with dual-mode support.

    Parameters
    ----------
    method_oracle : MethodOracle
        Method oracle for selecting compression methods per tensor.
    cascade_engine : DirectCascadeEngine or None
        Cascade engine for multi-stage compression. If None, uses single-method.
    memory_budget_mb : int
        Max memory for uncompressed data (default 4096 = 4 GB).
    mode : str
        ``'auto'`` (default), ``'streaming'``, or ``'ram'``.
    """

    def __init__(
        self,
        method_oracle: Any,
        cascade_engine: Optional[Any] = None,
        memory_budget_mb: int = 4096,
        mode: str = "auto",
    ) -> None:
        self._method_oracle = method_oracle
        self._cascade_engine = cascade_engine
        self._memory_budget_mb = memory_budget_mb
        self._mode_setting = mode
        self._peak_rss_mb: float = 0.0
        self._engine: Optional[Any] = None

    @property
    def _compression_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if hasattr(self._method_oracle, "_engine"):
            self._engine = self._method_oracle._engine
        elif hasattr(self._method_oracle, "engine"):
            self._engine = self._method_oracle.engine
        return self._engine

    def compress_model(
        self,
        model_path: str,
        output_path: str,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        progress_callback: Optional[
            Callable[[int, int, str, float, float], None]
        ] = None,
        quiet: bool = False,
        resume: bool = False,
    ) -> CompressionReport:
        total_start = time.perf_counter()

        tensor_info = self._header_only_scan(model_path)
        total_tensors = len(tensor_info)
        total_bytes = sum(nb for _, _, _, nb in tensor_info.values())
        model_gb = total_bytes / _GB
        avail_gb = check_available_ram_gb()

        if not quiet:
            print(f"Model: {total_tensors} tensors, {total_bytes / _GB:.1f} GB")
            print(
                f"System RAM: {avail_gb:.1f} GB available of "
                f"{check_total_ram_gb():.1f} GB total"
            )

        effective_mode = self._resolve_mode(model_path)

        if not quiet:
            print(
                f"Mode: {effective_mode.value.upper()} "
                f"(budget: {self._memory_budget_mb} MB)"
            )

        if effective_mode == CompressionMode.RAM:
            report = self._compress_in_ram(
                model_path=model_path,
                output_path=output_path,
                tensor_info=tensor_info,
                target_ratio=target_ratio,
                max_error=max_error,
                progress_callback=progress_callback,
                quiet=quiet,
            )
        else:
            report = self._compress_streaming(
                model_path=model_path,
                output_path=output_path,
                tensor_info=tensor_info,
                target_ratio=target_ratio,
                max_error=max_error,
                progress_callback=progress_callback,
                quiet=quiet,
                resume=resume,
            )

        report.mode = effective_mode.value
        report.model_size_gb = model_gb
        report.available_ram_gb = avail_gb
        report.memory_budget_mb = self._memory_budget_mb
        report.time_seconds = time.perf_counter() - total_start

        return report

    def _resolve_mode(self, model_path: str) -> CompressionMode:
        if self._mode_setting == "streaming":
            return CompressionMode.STREAMING
        if self._mode_setting == "ram":
            return CompressionMode.RAM
        return auto_detect_mode(model_path)

    def _compress_streaming(
        self,
        model_path: str,
        output_path: str,
        tensor_info: Dict[str, Tuple[tuple, str, int, int]],
        target_ratio: float,
        max_error: float,
        progress_callback: Optional[Callable] = None,
        quiet: bool = False,
        resume: bool = False,
    ) -> CompressionReport:
        total_tensors = len(tensor_info)
        report = CompressionReport()
        completed_tensors: set = set()

        checkpoint_path = output_path + ".streaming_checkpoint"
        if resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path) as f:
                    ckpt = json.load(f)
                completed_tensors = set(ckpt.get("completed_tensors", []))
                report.total_original_bytes = ckpt.get("total_orig_bytes", 0)
                report.total_compressed_bytes = ckpt.get("total_comp_bytes", 0)
                if not quiet:
                    print(
                        f"Resuming from checkpoint: {len(completed_tensors)} tensors done"
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load checkpoint: %s", e)

        writer = SSDWriter(
            output_path,
            metadata={
                "model": model_path,
                "streaming": True,
                "mode": "streaming",
                "target_ratio": target_ratio,
                "max_error": max_error,
            },
        )
        writer.__enter__()

        mmap_engine = MemoryMappedTensorEngine(model_path)
        errors: List[float] = []
        total_orig = report.total_original_bytes
        total_comp = report.total_compressed_bytes
        method_dist: Dict[str, int] = {}
        tensor_results: List[StreamedTensorResult] = []
        processed = len(completed_tensors)
        failures: List[str] = []

        try:
            sorted_names = sorted(
                tensor_info.keys(),
                key=lambda n: tensor_info[n][3],
            )

            if not quiet:
                print(f"\nCompressing {total_tensors} tensors (streaming)...")

            progress_handle = None
            task_id = None
            if HAS_RICH and not quiet:
                console = Console()
                progress_handle = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                )
                progress_handle.__enter__()
                task_id = progress_handle.add_task(
                    "Compressing (streaming)", total=total_tensors
                )

            for idx, name in enumerate(sorted_names):
                if name in completed_tensors:
                    continue

                t0 = time.perf_counter()
                shape, dtype_str, offset, nbytes = tensor_info[name]

                try:
                    tensor_view = mmap_engine.get_tensor(name)
                    tensor_data = np.asarray(tensor_view, dtype=np.float32)

                    data, meta, ratio_val, error_val = self._compress_with_oracle(
                        tensor=tensor_data,
                        name=name,
                        shape=shape,
                        target_ratio=target_ratio,
                        max_error=max_error,
                    )

                    method_str = meta.get("method", "unknown")
                    from spectralstream.format.compression import _name_to_method_id

                    method_id = _name_to_method_id(method_str)
                    writer.write_tensor_block(
                        compressed_data=data,
                        name=name,
                        shape=shape,
                        dtype=np.dtype(np.float32),
                        method_id=method_id,
                        params={
                            "original_shape": list(shape),
                            "original_dtype": dtype_str,
                            "compression_method": method_str,
                            "compression_params": meta,
                            "relative_error": error_val,
                            "compression_ratio": ratio_val,
                        },
                        quality_metrics={
                            "relative_error": error_val,
                            "compression_ratio": ratio_val,
                        },
                    )

                    total_orig += nbytes
                    total_comp += len(data)
                    method_dist[method_str] = method_dist.get(method_str, 0) + 1
                    errors.append(error_val)
                    processed += 1

                    tensor_result = StreamedTensorResult(
                        name=name,
                        original_shape=shape,
                        original_dtype=dtype_str,
                        original_nbytes=nbytes,
                        compressed_nbytes=len(data),
                        method=method_str,
                        compression_ratio=ratio_val,
                        relative_error=error_val,
                        time_seconds=time.perf_counter() - t0,
                        metadata=meta,
                    )
                    tensor_results.append(tensor_result)

                    if progress_callback:
                        progress_callback(
                            processed, total_tensors, name, ratio_val, error_val
                        )

                    mmap_engine.release_tensor(name)
                    del tensor_view, tensor_data, data, meta
                    gc.collect()

                    self._update_peak_mem()

                    if progress_handle and task_id is not None:
                        progress_handle.update(
                            task_id,
                            completed=processed,
                            description=f"[{method_str}] {name[-40:]} {ratio_val:.0f}x",
                        )

                    if processed % _CHECKPOINT_INTERVAL == 0:
                        completed_names = sorted_names[:processed]
                        self._save_checkpoint(
                            checkpoint_path, completed_names, total_orig, total_comp
                        )

                    if not quiet and processed % 10 == 0:
                        current_mem = self._get_current_rss_mb()
                        if current_mem > self._memory_budget_mb * 0.9:
                            logger.warning(
                                "Memory pressure: %.0f MB / %d MB budget",
                                current_mem,
                                self._memory_budget_mb,
                            )

                except Exception as e:
                    failures.append(name)
                    logger.error("Failed '%s': %s", name, e)
                    mmap_engine.release_tensor(name)
                    gc.collect()

            if progress_handle:
                progress_handle.__exit__(None, None, None)

            overall_ratio = max(total_orig / max(total_comp, 1), 1.0)
            avg_error = float(np.mean(errors)) if errors else 0.0
            max_err = float(np.max(errors)) if errors else 0.0

            report.total_tensors = total_tensors
            report.total_original_bytes = total_orig
            report.total_compressed_bytes = total_comp
            report.overall_ratio = overall_ratio
            report.avg_error = avg_error
            report.max_error = max_err
            report.peak_memory_mb = self._peak_rss_mb
            report.method_distribution = method_dist
            report.failures = failures
            report.tensor_results = tensor_results

        finally:
            writer.__exit__(None, None, None)
            mmap_engine.close()
            if not failures and os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except OSError:
                    pass

        if not quiet:
            print(
                f"\nStreaming complete: {total_tensors} tensors, "
                f"ratio={overall_ratio:.1f}x, mem={self._peak_rss_mb:.0f}MB"
            )

        return report

    def _compress_in_ram(
        self,
        model_path: str,
        output_path: str,
        tensor_info: Dict[str, Tuple[tuple, str, int, int]],
        target_ratio: float,
        max_error: float,
        progress_callback: Optional[Callable] = None,
        quiet: bool = False,
    ) -> CompressionReport:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total_tensors = len(tensor_info)
        report = CompressionReport()
        errors: List[float] = []
        total_orig = 0
        total_comp = 0
        method_dist: Dict[str, int] = {}
        tensor_results: List[StreamedTensorResult] = []
        failures: List[str] = []

        if not quiet:
            print(f"\nLoading {total_tensors} tensors into RAM...")

        tensors: Dict[str, np.ndarray] = {}
        load_t0 = time.perf_counter()

        from spectralstream.compression.engine._io import _SafetensorsIO

        io = _SafetensorsIO(use_mmap=False)
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            futures = {}
            for name, (shape, dtype_str, offset, nbytes) in tensor_info.items():
                future = pool.submit(
                    io.read, model_path, shape, dtype_str, offset, nbytes
                )
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    tensors[name] = future.result()
                except Exception as e:
                    failures.append(name)
                    logger.error("Failed to load '%s': %s", name, e)

        load_time = time.perf_counter() - load_t0
        if not quiet:
            print(f"Loaded {len(tensors)} tensors in {load_time:.1f}s")

        if not quiet:
            print(f"Compressing {len(tensors)} tensors (parallel)...")

        compress_t0 = time.perf_counter()
        compressed_results: Dict[str, Tuple[bytes, Dict[str, Any], float, float]] = {}

        num_workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {}
            for name, tensor in tensors.items():
                future = pool.submit(
                    self._compress_with_oracle,
                    tensor=tensor,
                    name=name,
                    shape=tensor_info[name][0],
                    target_ratio=target_ratio,
                    max_error=max_error,
                )
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    data, meta, ratio_val, error_val = future.result()
                    compressed_results[name] = (data, meta, ratio_val, error_val)
                except Exception as e:
                    failures.append(name)
                    logger.error("Compression failed '%s': %s", name, e)

            compress_time = time.perf_counter() - compress_t0

        if not quiet:
            print(f"Compressed in {compress_time:.1f}s")

        writer = SSDWriter(
            output_path,
            metadata={
                "model": model_path,
                "streaming": False,
                "mode": "ram",
                "target_ratio": target_ratio,
                "max_error": max_error,
            },
        )
        writer.__enter__()
        try:
            for name, (shape, dtype_str, offset, nbytes) in tensor_info.items():
                if name not in compressed_results:
                    continue
                data, meta, ratio_val, error_val = compressed_results[name]
                method_str = meta.get("method", "unknown")
                from spectralstream.format.compression import _name_to_method_id

                method_id = _name_to_method_id(method_str)
                writer.write_tensor_block(
                    compressed_data=data,
                    name=name,
                    shape=shape,
                    dtype=np.dtype(np.float32),
                    method_id=method_id,
                    params={
                        "original_shape": list(shape),
                        "original_dtype": dtype_str,
                        "compression_method": method_str,
                        "compression_params": meta,
                        "relative_error": error_val,
                        "compression_ratio": ratio_val,
                    },
                    quality_metrics={
                        "relative_error": error_val,
                        "compression_ratio": ratio_val,
                    },
                )

                total_orig += nbytes
                total_comp += len(data)
                method_dist[method_str] = method_dist.get(method_str, 0) + 1
                errors.append(error_val)

                tensor_results.append(
                    StreamedTensorResult(
                        name=name,
                        original_shape=shape,
                        original_dtype=dtype_str,
                        original_nbytes=nbytes,
                        compressed_nbytes=len(data),
                        method=method_str,
                        compression_ratio=ratio_val,
                        relative_error=error_val,
                        time_seconds=0.0,
                        metadata=meta,
                    )
                )

                if progress_callback:
                    progress_callback(
                        len(tensor_results), total_tensors, name, ratio_val, error_val
                    )

        finally:
            writer.__exit__(None, None, None)

        self._update_peak_mem()
        overall_ratio = max(total_orig / max(total_comp, 1), 1.0)
        avg_error = float(np.mean(errors)) if errors else 0.0
        max_err = float(np.max(errors)) if errors else 0.0

        report.total_tensors = total_tensors
        report.total_original_bytes = total_orig
        report.total_compressed_bytes = total_comp
        report.overall_ratio = overall_ratio
        report.avg_error = avg_error
        report.max_error = max_err
        report.peak_memory_mb = self._peak_rss_mb
        report.method_distribution = method_dist
        report.failures = failures
        report.tensor_results = tensor_results

        if not quiet:
            print(
                f"\nRAM-mode complete: {total_tensors} tensors, "
                f"ratio={overall_ratio:.1f}x, mem={self._peak_rss_mb:.0f}MB"
            )

        return report

    def _compress_with_oracle(
        self,
        tensor: np.ndarray,
        name: str,
        shape: Tuple[int, ...],
        target_ratio: float,
        max_error: float,
    ) -> Tuple[bytes, Dict[str, Any], float, float]:
        engine = self._compression_engine

        if (
            self._cascade_engine is not None
            and tensor.ndim >= 2
            and tensor.nbytes >= 1024
        ):
            try:
                tensor_type = (
                    _classify_by_name(name)
                    if hasattr(engine, "_classify_by_name")
                    else "weight"
                )
                data, meta = self._cascade_engine.execute_cascade(
                    engine, tensor, tensor_type, "balanced"
                )
                ratio_val = meta.get(
                    "total_ratio", float(tensor.nbytes / max(len(data), 1))
                )
                error_val = meta.get("total_error", 0.0)
                return data, meta, ratio_val, error_val
            except Exception as exc:
                logger.debug("Cascade failed for '%s', falling back: %s", name, exc)

        profile = engine.profiler.profile_tensor(tensor, name=name)
        error_budget = max_error / max(target_ratio, 1.0)
        methods = engine._select_methods(profile, error_budget, target_ratio)
        data, meta, ratio_val, error_val = engine.compress_tensor_with_validation(
            tensor, profile, methods, error_budget
        )
        return data, meta, ratio_val, error_val

    @staticmethod
    def _header_only_scan(
        model_path: str,
    ) -> Dict[str, Tuple[tuple, str, int, int]]:
        with open(model_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(_SAFETENSORS_HEADER_LEN))[0]
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes)

        data_start = _SAFETENSORS_HEADER_LEN + header_len
        tensor_info: Dict[str, Tuple[tuple, str, int, int]] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info.get("dtype", "F32")
            shape = tuple(info.get("shape", []))
            offsets = info.get("data_offsets", [0, 0])
            offset = data_start + offsets[0]
            nbytes = offsets[1] - offsets[0]
            tensor_info[name] = (shape, dtype_str, offset, nbytes)
        return tensor_info

    def _get_current_rss_mb(self) -> float:
        if not HAS_PSUTIL:
            return 0.0
        try:
            return psutil.Process().memory_info().rss / _MB
        except (OSError, AttributeError):
            return 0.0

    def _update_peak_mem(self) -> None:
        current = self._get_current_rss_mb()
        if current > self._peak_rss_mb:
            self._peak_rss_mb = current

    @staticmethod
    def _save_checkpoint(
        path: str,
        completed_names: List[str],
        total_orig: int,
        total_comp: int,
    ) -> None:
        ckpt = {
            "completed_tensors": completed_names,
            "total_orig_bytes": total_orig,
            "total_comp_bytes": total_comp,
            "timestamp": time.time(),
        }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(ckpt, f)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("Checkpoint save failed: %s", e)

    @staticmethod
    def format_report(report: CompressionReport) -> str:
        return "\n".join(report.summary_lines())

    @staticmethod
    def save_report_json(report: CompressionReport, path: str) -> None:
        data = {
            "mode": report.mode,
            "total_tensors": report.total_tensors,
            "total_original_bytes": report.total_original_bytes,
            "total_compressed_bytes": report.total_compressed_bytes,
            "overall_ratio": report.overall_ratio,
            "avg_error": report.avg_error,
            "max_error": report.max_error,
            "time_seconds": report.time_seconds,
            "peak_memory_mb": report.peak_memory_mb,
            "model_size_gb": report.model_size_gb,
            "available_ram_gb": report.available_ram_gb,
            "memory_budget_mb": report.memory_budget_mb,
            "method_distribution": report.method_distribution,
            "failures": report.failures,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


class UnifiedStreamingCompressionPipeline(UnifiedStreamingPipeline):
    """Backward-compatible wrapper that accepts ``engine`` + ``config``.

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        The compression engine.
    config : CompressionConfig, optional
        Compression configuration.
    memory_budget_mb : int
        Memory budget in MB (default 4096).
    """

    def __init__(
        self,
        engine: Any,
        config: Optional[CompressionConfig] = None,
        memory_budget_mb: int = 4096,
    ) -> None:
        self._cfg = config if config is not None else CompressionConfig()
        cascade_engine: Optional[Any] = None
        if hasattr(engine, "oracle"):
            method_oracle = engine.oracle
        elif hasattr(engine, "method_oracle"):
            method_oracle = engine.method_oracle
        else:
            method_oracle = engine

        if hasattr(engine, "quantum_cascade"):
            cascade_engine = engine.quantum_cascade
        elif DirectCascadeEngine is not None:
            try:
                cascade_engine = DirectCascadeEngine(store_all_stages=True)
            except Exception:
                cascade_engine = None

        super().__init__(
            method_oracle=method_oracle,
            cascade_engine=cascade_engine,
            memory_budget_mb=memory_budget_mb,
            mode="auto",
        )
        self._engine_ref = engine
        self._config_ref = self._cfg

    def compress_model(
        self,
        model_path: str,
        output_path: str,
        mode: Optional[CompressionMode] = None,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        progress_callback: Optional[Callable] = None,
        quiet: bool = False,
        resume: bool = False,
    ) -> CompressionReport:
        if mode is not None:
            self._mode_setting = (
                mode.value if isinstance(mode, CompressionMode) else mode
            )
        return super().compress_model(
            model_path=model_path,
            output_path=output_path,
            target_ratio=target_ratio,
            max_error=max_error,
            progress_callback=progress_callback,
            quiet=quiet,
            resume=resume,
        )

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.profiler.scanner import ModelScanner, ModelScanResult
from spectralstream.compression.profiler.analyzer import (
    SensitivityAnalyzer,
    TensorProfile,
)
from spectralstream.compression.profiler.allocator import (
    BitAllocationOptimizer,
    BitAllocation,
)
from spectralstream.compression.profiler.report import ReportBuilder, ProfilingReport
from spectralstream.compression.profiler.calibration import CalibrationData

logger = logging.getLogger(__name__)


class ProfilerEngine:
    def __init__(self, max_sample_elements: int = 100_000) -> None:
        self.scanner = ModelScanner()
        self.analyzer = SensitivityAnalyzer(max_sample_elements)
        self.allocator = BitAllocationOptimizer()
        self._lock = threading.Lock()

    def profile_model(
        self,
        model_path: str,
        target_ratio: float = 100.0,
        max_error: float = 0.02,
        optimize: str = "min_error",
        calibration_data: Optional[CalibrationData] = None,
    ) -> ProfilingReport:
        t_start = time.perf_counter()
        logger.info(
            "Profiling model: %s (ratio=%.1f, error=%.6f)",
            model_path,
            target_ratio,
            max_error,
        )
        scan = self.scanner.scan(model_path)
        logger.info(
            "Scanned %d tensors, %.1f MB total",
            scan.tensor_count,
            scan.total_bytes / 1024 / 1024,
        )
        profiles: Dict[str, TensorProfile] = {}
        for name, (shape, dtype_str, offset, nbytes) in scan.tensors.items():
            tensor = self._read_tensor(
                model_path, scan.format, shape, dtype_str, offset, nbytes
            )
            profile = self.analyzer.profile(tensor, name=name)
            if calibration_data is not None:
                adj = calibration_data.get_adjusted_sensitivity(
                    profile.sensitivity, name
                )
                profile.sensitivity = adj
                profile.sensitivity_category = (
                    "HIGH" if adj >= 0.8 else ("MEDIUM" if adj >= 0.5 else "LOW")
                )
            profiles[name] = profile
            logger.debug(
                "Profiled %s: type=%s, method=%s, sens=%.3f",
                name,
                profile.tensor_type,
                profile.recommended_method,
                profile.sensitivity,
            )
        allocations = self.allocator.allocate(
            profiles, target_ratio, max_error, optimize
        )
        heatmap = self.analyzer.generate_heatmap(profiles)
        elapsed = time.perf_counter() - t_start
        report = ReportBuilder.build(scan, profiles, allocations, heatmap, elapsed)
        logger.info(
            "Profiling complete: %d tensors in %.2fs", scan.tensor_count, elapsed
        )
        return report

    def _read_tensor(
        self,
        path: str,
        fmt: str,
        shape: Tuple[int, ...],
        dtype_str: str,
        offset: int,
        nbytes: int,
    ) -> np.ndarray:
        if fmt == "safetensors":
            return self._read_safetensor(path, shape, dtype_str, offset, nbytes)
        if fmt == "gguf":
            return self._read_gguf(path, offset, nbytes)
        if fmt == "ssf":
            return self._read_ssf(path, offset, nbytes)
        raise ValueError(f"Unknown format: {fmt}")

    @staticmethod
    def _read_safetensor(
        path: str, shape: Tuple[int, ...], dtype_str: str, offset: int, nbytes: int
    ) -> np.ndarray:
        if dtype_str == "BF16":
            # BF16: read as uint16, convert to uint32, left-pad with 16 zero bits, view as float32
            with open(path, "rb") as f:
                f.seek(offset)
                raw = f.read(nbytes)
            tensor_u16 = np.frombuffer(raw, dtype=np.uint16)
            tensor = tensor_u16.astype(np.uint32) << 16
            tensor = tensor.view(np.float32)
            if shape:
                tensor = tensor.reshape(shape)
            return tensor
        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "I64": np.int64,
            "I32": np.int32,
            "I16": np.int16,
            "I8": np.int8,
            "U8": np.uint8,
        }
        np_dtype = dtype_map.get(dtype_str, np.float32)
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(nbytes)
        tensor = np.frombuffer(raw, dtype=np_dtype)
        return (
            tensor.reshape(shape).astype(np.float32)
            if shape
            else tensor.astype(np.float32)
        )

    @staticmethod
    def _read_gguf(path: str, offset: int, nbytes: int) -> np.ndarray:
        try:
            from spectralstream.format.gguf_parser_engine import GGMLDequantizer

            with open(path, "rb") as f:
                f.seek(offset)
                raw = f.read(nbytes)
            return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        except ImportError:
            with open(path, "rb") as f:
                f.seek(offset)
                raw = f.read(nbytes)
            return np.frombuffer(raw, dtype=np.float32)

    @staticmethod
    def _read_ssf(path: str, offset: int, nbytes: int) -> np.ndarray:
        try:
            from spectralstream.format.ssf_format import SSFReader

            reader = SSFReader(path, mmap_mode=True)
            ssf_index = reader._index
            name: Optional[str] = None
            if ssf_index is not None:
                for e in ssf_index:
                    if e.data_offset == offset:
                        name = e.name
                        break
            if name is not None:
                result = reader.get_tensor(name)
                reader.close()
                return result
            with open(path, "rb") as f:
                f.seek(offset)
                raw = f.read(nbytes)
            reader.close()
            return np.frombuffer(raw, dtype=np.float32)
        except ImportError:
            with open(path, "rb") as f:
                f.seek(offset)
                raw = f.read(nbytes)
            return np.frombuffer(raw, dtype=np.float32)

    def profile_tensor(self, tensor: np.ndarray, name: str = "") -> TensorProfile:
        return self.analyzer.profile(tensor, name)

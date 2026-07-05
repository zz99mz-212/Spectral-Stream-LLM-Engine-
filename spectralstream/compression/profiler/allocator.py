from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.profiler.analyzer import TensorProfile


@dataclass
class BitAllocation:
    tensor: str
    bits: int
    method: str
    expected_error: float
    expected_ratio: float
    allocation_weight: float


class BitAllocationOptimizer:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def allocate(
        self,
        profiles: Dict[str, TensorProfile],
        target_ratio: float,
        max_error: float = 0.02,
        optimize: str = "min_error",
    ) -> List[BitAllocation]:
        if not profiles:
            return []
        if optimize == "min_error":
            return self._allocate_min_error(profiles, target_ratio, max_error)
        return self._allocate_max_ratio(profiles, target_ratio, max_error)

    def _allocate_min_error(
        self, profiles: Dict[str, TensorProfile], target_ratio: float, max_error: float
    ) -> List[BitAllocation]:
        raw_weights = {name: max(p.sensitivity, 1e-6) for name, p in profiles.items()}
        w_min = min(raw_weights.values())
        w_max = max(raw_weights.values())
        w_range = max(w_max - w_min, 1.0)
        total_bytes = sum(p.nbytes for p in profiles.values())
        target_compressed = total_bytes / max(target_ratio, 1.0)
        allocations: List[BitAllocation] = []
        for name, p in profiles.items():
            w_norm = (raw_weights[name] - w_min) / w_range
            weight = 1.0 - w_norm * 0.8
            allocated_bytes = (
                target_compressed * (p.nbytes / max(total_bytes, 1)) * weight
            )
            bits = self._bytes_to_bits(allocated_bytes, p.n_elements, p.dtype)
            bits = max(2, min(bits, 16))
            method, exp_err, exp_ratio = self._method_for_bits(bits, p, max_error)
            allocations.append(
                BitAllocation(
                    tensor=name,
                    bits=bits,
                    method=method,
                    expected_error=exp_err,
                    expected_ratio=exp_ratio,
                    allocation_weight=weight,
                )
            )
        return allocations

    def _allocate_max_ratio(
        self, profiles: Dict[str, TensorProfile], target_ratio: float, max_error: float
    ) -> List[BitAllocation]:
        total_bytes = sum(p.nbytes for p in profiles.values())
        target_compressed = total_bytes / max(target_ratio, 1.0)
        allocations: List[BitAllocation] = []
        remaining_compressed = target_compressed
        sorted_profiles = sorted(
            profiles.items(), key=lambda x: (x[1].sensitivity, -x[1].nbytes)
        )
        for name, p in sorted_profiles:
            bits = self._bits_for_error(p, max_error)
            bits = max(2, min(bits, 16))
            bytes_per_elem = bits / 8
            if isinstance(p.dtype, str) and "float" in p.dtype.lower():
                bytes_per_elem *= 4 / max(np.finfo(np.float32).bits / 8, 1)
            expected_compressed = p.n_elements * bytes_per_elem
            if expected_compressed > remaining_compressed / len(sorted_profiles) * 2:
                bits = max(2, bits - 2)
            method, exp_err, exp_ratio = self._method_for_bits(bits, p, max_error)
            allocations.append(
                BitAllocation(
                    tensor=name,
                    bits=bits,
                    method=method,
                    expected_error=exp_err,
                    expected_ratio=exp_ratio,
                    allocation_weight=1.0,
                )
            )
        return allocations

    @staticmethod
    def _bytes_to_bits(allocated_bytes: float, n_elements: int, dtype_str: str) -> int:
        if n_elements <= 0:
            return 8
        return int(round(allocated_bytes / n_elements * 8))

    @staticmethod
    def _bits_for_error(p: TensorProfile, max_error: float) -> int:
        if p.std < 1e-6:
            return 4
        if p.outlier_ratio > 0.3:
            return 4 if max_error > 0.01 else 8
        bits_map = {0.05: 4, 0.02: 6, 0.01: 8, 0.005: 12, 0.001: 16}
        for threshold, bits in sorted(bits_map.items()):
            if max_error >= threshold:
                return bits
        return 16

    @staticmethod
    def _method_for_bits(
        bits: int, p: TensorProfile, max_error: float
    ) -> Tuple[str, float, float]:
        if bits >= 16:
            return "passthrough", 0.0, 1.0
        if bits >= 12:
            return "block_int8", 0.001, 3.0
        if bits >= 8:
            if p.outlier_ratio > 0.3:
                return "sparsity_int4", 0.015, 6.0
            if p.energy_concentration > 0.8:
                return "hadamard_int8", 0.005, 4.0
            return "block_int8", 0.005, 4.0
        if bits >= 6:
            return "block_int4", 0.02, 6.0
        if bits >= 4:
            return "hadamard_int4", 0.04, 8.0
        return "binary_quant", 0.15, 15.0

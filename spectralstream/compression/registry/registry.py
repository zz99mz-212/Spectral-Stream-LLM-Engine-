from __future__ import annotations

from typing import Any, Dict, List, Optional

from spectralstream.compression.registry.enum import CompressionMethod
from spectralstream.compression.registry.metadata import MethodMetadata


class MethodRegistry:
    """Global registry of ALL compression methods with metadata."""

    _methods: Dict[CompressionMethod, MethodMetadata] = {}

    @classmethod
    def register(cls, method_id: CompressionMethod, metadata: MethodMetadata) -> None:
        cls._methods[method_id] = metadata

    @classmethod
    def get(cls, method_id: CompressionMethod) -> Optional[MethodMetadata]:
        return cls._methods.get(method_id)

    @classmethod
    def list_methods(cls, category: Optional[str] = None) -> List[MethodMetadata]:
        if category is None:
            return list(cls._methods.values())
        return [m for m in cls._methods.values() if m.category == category]

    @classmethod
    def list_categories(cls) -> List[str]:
        return sorted({m.category for m in cls._methods.values()})

    @classmethod
    def methods_for_tensor_type(cls, tensor_type: str) -> List[CompressionMethod]:
        tensor_type = tensor_type.lower()
        candidates = []
        for method_id, meta in cls._methods.items():
            name_lower = meta.name.lower()
            if tensor_type == "sparse" and (
                "sparsity" in name_lower or "sparse" in name_lower
            ):
                candidates.append(method_id)
            elif tensor_type == "low_rank" and any(
                t in name_lower for t in ("svd", "tt_", "tucker", "cp_", "low_rank")
            ):
                candidates.append(method_id)
            elif tensor_type == "dense" and (
                "block" in name_lower or "hadamard" in name_lower or "int" in name_lower
            ):
                candidates.append(method_id)
            elif tensor_type == "structured" and any(
                t in name_lower
                for t in ("toeplitz", "circulant", "hankel", "butterfly", "monarch")
            ):
                candidates.append(method_id)
        return candidates

    @classmethod
    def get_best_method(
        cls, profile: Any, target_ratio: float, max_error: float
    ) -> CompressionMethod:
        candidates = []
        for method_id, meta in cls._methods.items():
            min_r, max_r = meta.compression_ratio_range
            min_e, max_e = meta.expected_error_range
            if max_r >= target_ratio * 0.7 and min_e <= max_error:
                score = min(target_ratio / max(min_r, 1), 2.0) - (
                    max_e / max(max_error, 1e-10)
                )
                candidates.append((score, method_id))
        if not candidates:
            candidates = [(0.0, mid) for mid in cls._methods]
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

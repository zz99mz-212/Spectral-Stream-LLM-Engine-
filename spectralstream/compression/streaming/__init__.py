"""
Unified Streaming Compression — dual-mode support.

Mode 1: STREAMING_FROM_DISK — memory-map safetensors, process tensor-by-tensor,
         flush compressed output immediately.  Peak ~1-2 tensor weights + working mem.
         For 365GB model on 64GB RAM (~8-16GB peak).

Mode 2: IN_RAM — load all tensors, compress in parallel, write at end.
         Faster but requires model <= 50% of available RAM.

Auto-detects mode based on model size vs available system RAM.
"""

from __future__ import annotations

from .unified_streaming_pipeline import (
    UnifiedStreamingCompressionPipeline,
    CompressionMode,
    auto_detect_mode,
    check_available_ram_gb,
    check_model_size_gb,
)

__all__ = [
    "UnifiedStreamingCompressionPipeline",
    "CompressionMode",
    "auto_detect_mode",
    "check_available_ram_gb",
    "check_model_size_gb",
]

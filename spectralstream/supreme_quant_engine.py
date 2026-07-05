"""
Backward compatibility stub — the real engine is in spectralstream.compression.engine
"""

from __future__ import annotations
from spectralstream.compression.engine import (  # noqa: F401
    CompressedTensor as CompressedWeight,
    CompressionConfig as CompressionBudget,
    CompressionIntelligenceEngine as SupremeQuantEngine,
    TensorProfile,
    CompressionProfiler as TensorProfiler,
)
from spectralstream.compression.engine._dataclasses import (
    CompressedTensor,
    CompressionConfig,
    CompressionReport,
)

CompressionPipeline = CompressionIntelligenceEngine = SupremeQuantEngine
MethodLibrary = None

compress = CompressionIntelligenceEngine
decompress = None
get_engine = None
profile = None

__all__ = [
    "CompressedWeight",
    "CompressionBudget",
    "CompressionPipeline",
    "MethodLibrary",
    "SupremeQuantEngine",
    "TensorProfiler",
    "TensorProfile",
    "compress",
    "decompress",
    "get_engine",
    "profile",
]

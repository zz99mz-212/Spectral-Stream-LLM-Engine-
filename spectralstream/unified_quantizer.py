"""
Backward compatibility re-export — moved to spectralstream.compression.unified_quantizer.
"""

from __future__ import annotations
import warnings as _warnings

_warnings.warn(
    "spectralstream.unified_quantizer is deprecated. "
    "Use spectralstream.compression.unified_quantizer instead.",
    DeprecationWarning,
    stacklevel=2,
)

_imported_names: dict = {}

for _name in [
    "CompressionPipeline2000",
    "CompressionPipeline2000Config",
    "UnifiedQuantizer",
    "QAOABitAllocator",
    "StabilizerQuantizer",
    "PredictiveCodingQuantizer",
    "TernaryWeightQuantizer",
    "SpectralSparsification",
    "HierarchicalMPSCompressor",
]:
    try:
        _mod = __import__(
            "spectralstream.compression.unified_quantizer",
            fromlist=[_name],
        )
        _imported_names[_name] = getattr(_mod, _name)
    except (ImportError, AttributeError):
        _imported_names[_name] = None  # type: ignore

CompressionPipeline2000 = _imported_names["CompressionPipeline2000"]
CompressionPipeline2000Config = _imported_names["CompressionPipeline2000Config"]
UnifiedQuantizer = _imported_names["UnifiedQuantizer"]
QAOABitAllocator = _imported_names["QAOABitAllocator"]
StabilizerQuantizer = _imported_names["StabilizerQuantizer"]
PredictiveCodingQuantizer = _imported_names["PredictiveCodingQuantizer"]
TernaryWeightQuantizer = _imported_names["TernaryWeightQuantizer"]
SpectralSparsification = _imported_names["SpectralSparsification"]
HierarchicalMPSCompressor = _imported_names["HierarchicalMPSCompressor"]

# Backward compat alias
Pipeline2000Config = CompressionPipeline2000Config

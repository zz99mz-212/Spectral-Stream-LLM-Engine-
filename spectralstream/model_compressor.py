"""
Backward compatibility stub — moved to spectralstream.compression.model_compressor
"""

from __future__ import annotations
import warnings as _warnings

_warnings.warn(
    "spectralstream.model_compressor moved to spectralstream.compression.model_compressor",
    DeprecationWarning,
    stacklevel=2,
)
from spectralstream.compression.model_compressor import (  # noqa: F401,F403
    CompressionReport,
    CompressionResult,
    ModelCompressor,
    TensorProfile,
    ValidationReport,
    _compress_dct_block,
    _compress_int4,
    _compress_int8,
    _decompress_dct_block,
    _decompress_int4,
    _decompress_int8,
    _detect_layer_id,
    _error_metrics,
    _get_sensitivity,
    _is_embedding,
    _zigzag_scan,
    _zigzag_unscan,
)

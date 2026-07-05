"""
Backward compatibility re-export — moved to spectralstream.format.gguf_parser_engine
"""

from __future__ import annotations
import warnings as _warnings

_warnings.warn(
    "spectralstream.gguf_parser_engine is deprecated. Use spectralstream.format.gguf_parser_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

from spectralstream.format.gguf_parser_engine import (  # noqa: F401,F403
    GGMLDequantizer,
    GGML_BLOCK_SIZE,
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_NAMES,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q8_K,
    GGML_TYPE_TQ2_0,
    GGUFModelPatcher,
    GGUFParser,
    GGUFParserEngine,
    GGUF_MAGIC,
    GGUF_VERSION_V2,
    GGUF_VERSION_V3,
    MMAPWeightLoader,
    PredictiveWeightPrefetcher,
    ResonantWeightLoader,
    SpectralDequantizer,
    SpectralTensorConverter,
    WeightCache,
    ZeroCopySpectralEngine,
    validate_gguf,
)

"""Utility functions for the compression engine."""

import json
from typing import Any, Optional

from ._dataclasses import CompressionConfig
from ._orchestrator import CompressionIntelligenceEngine


def create_engine(
    config: Optional[CompressionConfig] = None,
) -> CompressionIntelligenceEngine:
    return CompressionIntelligenceEngine(config)


def estimate_swift_ratio(config: CompressionConfig) -> float:
    return max(config.target_ratio, config.min_ratio)


def load_compression_config(path: str) -> CompressionConfig:
    with open(path) as f:
        raw = json.load(f)
    return CompressionConfig(**raw)


def compression_config_from_ss_config(ss_config: Any) -> CompressionConfig:
    return CompressionConfig(
        target_ratio=getattr(ss_config.spectral, "kv_compression", 5000.0),
        num_workers=getattr(ss_config.hardware, "cpu_cores", 4),
    )

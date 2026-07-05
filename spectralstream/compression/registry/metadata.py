from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from spectralstream.compression.registry.enum import CompressionMethod


@dataclass
class MethodMetadata:
    """Rich metadata for a compression method."""

    method_id: CompressionMethod
    name: str
    category: str
    description: str
    compression_ratio_range: Tuple[float, float]
    expected_error_range: Tuple[float, float]
    preserves_precision: bool = False
    supports_streaming: bool = False
    requires_calibration: bool = False
    is_lossless: bool = False

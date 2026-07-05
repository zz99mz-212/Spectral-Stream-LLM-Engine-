"""Compression method registry — backward-compatible wrapper.

Auto-registers all methods from METHOD_CLASSES on import.
"""

from __future__ import annotations

from spectralstream.compression.registry import (
    CompressionMethod,
    MethodMetadata,
    MethodRegistry,
    _register_all,
)

# Register ALL methods from METHOD_CLASSES
_register_all()

__all__ = [
    "CompressionMethod",
    "MethodMetadata",
    "MethodRegistry",
    "_register_all",
]

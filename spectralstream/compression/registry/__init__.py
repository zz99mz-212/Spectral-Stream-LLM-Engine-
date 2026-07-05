from spectralstream.compression.registry.enum import CompressionMethod
from spectralstream.compression.registry.metadata import MethodMetadata
from spectralstream.compression.registry.registry import MethodRegistry
from spectralstream.compression.registry.registration import _register_all

# Auto-register ALL methods from METHOD_CLASSES on import
_register_all()

__all__ = [
    "CompressionMethod",
    "MethodMetadata",
    "MethodRegistry",
    "_register_all",
]

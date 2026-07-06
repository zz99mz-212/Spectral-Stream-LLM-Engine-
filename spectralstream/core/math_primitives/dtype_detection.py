"""Dynamic dtype detection and handling for model weights.

Reads safetensors/SSF headers to determine the native dtype
of every tensor and ensures the engine handles it correctly.
Supports BF16, F32, F16, F8_E4M3, F8_E5M2, and integer types.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Map from safetensors dtype strings to numpy dtypes
SAFETENSORS_DTYPE_MAP: Dict[str, np.dtype] = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "bfloat16": np.uint16,
    "bf16": np.uint16,
    "F8_E4M3": np.uint8,
    "F8_E5M2": np.uint8,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
}

# Which dtypes are floating-point types in safetensors
SAFETENSORS_FLOAT_DTYPES = {
    "F32",
    "F16",
    "BF16",
    "bfloat16",
    "bf16",
    "F8_E4M3",
    "F8_E5M2",
}

# Which safetensors dtypes store bfloat16
BFLOAT16_MARKERS = {"BF16", "bfloat16", "bf16"}

# Canonical name mapping
DTYPE_CANONICAL: Dict[str, str] = {
    "BF16": "bfloat16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "F32": "float32",
    "F16": "float16",
    "F8_E4M3": "float8_e4m3",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
}

NDARRAY_DTYPE_TO_SAFETENSORS: Dict[np.dtype, str] = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.int64): "I64",
    np.dtype(np.int32): "I32",
    np.dtype(np.int16): "I16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
    np.dtype(np.uint16): "U16",
    np.dtype(np.uint32): "U32",
    np.dtype(np.uint64): "U64",
}


def normalize_dtype(dtype_str: str) -> str:
    """Normalize a dtype string to canonical safetensors format."""
    return DTYPE_CANONICAL.get(dtype_str, dtype_str.lower())


def detect_native_dtype(tensor: np.ndarray, hint: str = "") -> str:
    """Detect the native dtype of a tensor loaded from safetensors.

    Uses a priority chain:
    1.  ``hint`` string from safetensors header (e.g. "BF16", "F32")
    2.  If no hint: infer from numpy dtype, with special handling for
        bfloat16 (stored as uint16 in safetensors).

    Returns
    -------
    str
        One of: 'bfloat16', 'float16', 'float32', 'float64', 'float8_e4m3',
        'float8_e5m2', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
        'uint32', 'uint64'.
    """
    if hint:
        canonical = normalize_dtype(hint)
        if canonical != hint.lower():
            return canonical
        return hint.lower()

    # No hint — infer from numpy dtype
    if tensor.dtype == np.uint16:
        return "bfloat16"
    if tensor.dtype == np.uint8:
        return "uint8"
    return str(tensor.dtype)


def get_dtype_size(dtype_str: str) -> float:
    """Get bytes per element for a dtype string."""
    sizes: Dict[str, float] = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
        "float64": 8,
        "float8_e4m3": 1,
        "float8_e5m2": 1,
        "int8": 1,
        "int4": 0.5,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "uint8": 1,
        "uint16": 2,
        "uint32": 4,
        "uint64": 8,
    }
    return sizes.get(dtype_str, 4)


def get_precision_bits(dtype_str: str) -> int:
    """Get precision bits for a dtype."""
    bits: Dict[str, int] = {
        "bfloat16": 16,
        "float16": 16,
        "float32": 32,
        "float64": 64,
        "float8_e4m3": 8,
        "float8_e5m2": 8,
        "int8": 8,
        "int4": 4,
        "int16": 16,
        "int32": 32,
        "int64": 64,
        "uint8": 8,
        "uint16": 16,
        "uint32": 32,
        "uint64": 64,
    }
    return bits.get(dtype_str, 32)


def dtype_is_float(dtype_str: str) -> bool:
    """Check if a dtype string represents a floating-point type."""
    return dtype_str.upper() in {
        "F32",
        "F16",
        "BF16",
        "F8_E4M3",
        "F8_E5M2",
    } or dtype_str.lower() in {
        "float32",
        "float16",
        "bfloat16",
        "float8_e4m3",
        "float8_e5m2",
    }


def dtype_is_bf16(dtype_str: str) -> bool:
    """Check if a dtype string is bfloat16."""
    return dtype_str.upper() == "BF16" or dtype_str.lower() in ("bfloat16", "bf16")


def safetensors_dtype_to_str(st_dtype: str) -> str:
    """Convert safetensors dtype string to canonical form.

    Examples
    --------
    >>> safetensors_dtype_to_str("BF16")
    'bfloat16'
    >>> safetensors_dtype_to_str("F32")
    'float32'
    >>> safetensors_dtype_to_str("F8_E4M3")
    'float8_e4m3'
    """
    return normalize_dtype(st_dtype)


def ndarray_dtype_to_safetensors(dt: np.dtype) -> str:
    """Convert a numpy dtype to safetensors dtype string."""
    return NDARRAY_DTYPE_TO_SAFETENSORS.get(dt, "F32")


def normalize_for_compression(
    tensor: np.ndarray, hint: str = ""
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize a tensor for compression, tracking its original dtype.

    Converts to float32 for computation while preserving original dtype info.
    BF16 (uint16) tensors are converted via bit manipulation.
    FP8 (uint8) tensors are converted to float32.
    Integer tensors are converted to float32.

    Returns
    -------
    tuple of (normalized_tensor, metadata)
        metadata includes:
        - 'native_dtype': original dtype string
        - 'native_dtype_size': bytes per element
        - 'native_precision_bits': precision bits
        - 'dtype_conversion': whether conversion occurred
        - 'dtype_conversion_error': max error from conversion (if any)
    """
    native = detect_native_dtype(tensor, hint)
    meta: Dict[str, Any] = {
        "native_dtype": native,
        "native_dtype_size": get_dtype_size(native),
        "native_precision_bits": get_precision_bits(native),
        "dtype_conversion": False,
        "dtype_conversion_error": 0.0,
    }

    if native == "bfloat16":
        from .bfloat16 import bfloat16_to_float32

        f32 = bfloat16_to_float32(tensor)
        meta["dtype_conversion"] = True
        from .bfloat16 import float32_to_bfloat16

        back = float32_to_bfloat16(f32)
        error = float(
            np.max(np.abs(tensor.astype(np.float64) - back.astype(np.float64)))
        )
        meta["dtype_conversion_error"] = error
        return f32, meta

    if tensor.dtype not in (np.float32, np.float64):
        f32 = tensor.astype(np.float32)
        meta["dtype_conversion"] = True
        error = float(
            np.max(np.abs(tensor.astype(np.float64) - f32.astype(np.float64)))
        )
        meta["dtype_conversion_error"] = error
        return f32, meta

    return tensor.astype(np.float32), meta


def denormalize_from_compression(
    tensor: np.ndarray, meta: Dict[str, Any]
) -> np.ndarray:
    """Convert a compressed/reconstructed tensor back to its native dtype.

    If the original was bfloat16, convert back to uint16 bfloat16 format.
    If the original was float16, convert back.
    Otherwise return as float32.

    Parameters
    ----------
    tensor : np.ndarray
        Float32 tensor from decompression.
    meta : dict
        Metadata from normalize_for_compression().

    Returns
    -------
    np.ndarray
        Tensor in its original dtype.
    """
    native = meta.get("native_dtype", "float32")

    if dtype_is_bf16(native):
        from .bfloat16 import float32_to_bfloat16

        return float32_to_bfloat16(tensor)
    if native == "float16":
        return tensor.astype(np.float16)
    if native in ("int8", "int4"):
        return tensor.astype(np.int8)
    return tensor.astype(np.float32)


def scan_safetensors_header(path: str) -> Dict[str, Any]:
    """Read and parse a safetensors file header, returning the full JSON.

    Parameters
    ----------
    path : str
        Path to .safetensors file.

    Returns
    -------
    dict
        Parsed JSON header with tensor name -> {dtype, shape, data_offsets}.
    """
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        import json

        header = json.loads(f.read(header_len))
    return header


def analyze_model_dtypes(path: str) -> Dict[str, Any]:
    """Analyze all dtypes in a safetensors model file.

    Parameters
    ----------
    path : str
        Path to .safetensors file.

    Returns
    -------
    dict with keys:
        - 'dtype_set': set of dtype strings used
        - 'count_by_dtype': dict of dtype -> count
        - 'total_tensors': total number of tensors
        - 'mixed': whether model uses multiple dtypes
    """
    header = scan_safetensors_header(path)
    dtypes: Dict[str, int] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = info.get("dtype", "F32")
        dtypes[dt] = dtypes.get(dt, 0) + 1
    return {
        "dtype_set": set(dtypes.keys()),
        "count_by_dtype": dtypes,
        "total_tensors": sum(dtypes.values()),
        "mixed": len(dtypes) > 1,
    }

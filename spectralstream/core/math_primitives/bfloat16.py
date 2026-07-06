"""bfloat16 utilities: detection, conversion, memory-efficient storage.

BF16 values are stored as uint16 in memory (the native safetensors format).
Only convert to float32 when computation requires it, and convert back
to uint16 for storage to halve memory usage vs. float32.
"""

import numpy as np


def is_bfloat16(arr: np.ndarray) -> bool:
    """Check if a numpy array represents bfloat16 data (stored as uint16).

    A uint16 array with the bfloat16 flags in metadata is the canonical
    representation.  Without metadata, a uint16 array *may* be bfloat16 —
    we assume it is when a dtype string field or explicit flag says so.
    For bare uint16 arrays without context, we conservatively return False.
    """
    return arr.dtype == np.uint16


def bfloat16_to_float32(arr: np.ndarray) -> np.ndarray:
    """Convert BF16 (stored as uint16) to float32 via bit manipulation.

    BF16 is the upper 16 bits of float32.  The conversion is:
    uint16 value → uint32 << 16 → view as float32.
    """
    if arr.dtype != np.uint16:
        raise TypeError(f"Expected uint16 (BF16), got {arr.dtype}")
    return (arr.astype(np.uint32) << 16).view(np.float32)


def float32_to_bfloat16(arr: np.ndarray, round_to_even: bool = True) -> np.ndarray:
    """Convert float32 to BF16 stored as uint16.

    Uses round-to-nearest-even for highest precision preservation.
    Simply truncating the lower 16 bits loses ~1 ULP of precision.

    Parameters
    ----------
    arr : np.ndarray
        Float32 array to convert.
    round_to_even : bool
        If True (default), use round-to-nearest-even.  If False, truncate.

    Returns
    -------
    np.ndarray (uint16)
        BF16 values in uint16 storage.
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    arr_view = arr.view(np.uint32)
    if round_to_even:
        rounding = 0x7FFF + ((arr_view >> 16) & 1)
        return ((arr_view + rounding) >> 16).astype(np.uint16)
    else:
        return (arr_view >> 16).astype(np.uint16)


def ensure_float32(arr: np.ndarray) -> np.ndarray:
    """Convert to float32 for computation.

    If arr is uint16 (BF16), convert to float32.
    Otherwise, convert to float32 if not already.
    Returns a new array — caller must free the original to save memory.
    """
    if arr.dtype == np.uint16:
        return bfloat16_to_float32(arr)
    return arr.astype(np.float32, copy=False)


def maybe_contract_to_uint16(arr: np.ndarray, input_was_bf16: bool) -> np.ndarray:
    """Convert float32 result back to BF16 (uint16) if input was BF16.

    If input_was_bf16 is False, returns arr unchanged (as float32).
    """
    if input_was_bf16:
        return float32_to_bfloat16(arr)
    return arr


def compression_ratio_adjustment(native_dtype: str) -> float:
    """Return the factor by which the compression ratio should be adjusted.

    BF16 tensors are half the size of float32, so the compression ratio
    computed from tensor.nbytes is 2x smaller than the effective ratio.
    """
    if native_dtype.lower() in ("bfloat16", "bf16"):
        return 0.5
    return 1.0


def dtype_is_bf16(dtype_str: str) -> bool:
    """Check if a dtype string is bfloat16."""
    return dtype_str.upper() == "BF16" or dtype_str.lower() in ("bfloat16", "bf16")


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

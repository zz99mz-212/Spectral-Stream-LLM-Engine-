import numpy as np
from typing import Optional
from spectralstream.core.math_primitives import (
    is_bfloat16,
    bfloat16_to_float32,
    float32_to_bfloat16,
)

_STORAGE_EFFICIENCY = {
    np.dtype("uint16"): 2,
    np.dtype("float16"): 2,
    np.dtype("float32"): 4,
    np.dtype("float64"): 8,
}

_DTYPE_CODE = {
    np.dtype("float16"): 0,
    np.dtype("uint16"): 1,
    np.dtype("float32"): 2,
    np.dtype("float64"): 3,
}

_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}


def detect_storage_dtype(
    tensor: np.ndarray, force_precision: Optional[str] = None
) -> np.dtype:
    if force_precision is not None:
        return np.dtype(force_precision)
    dt = tensor.dtype
    if is_bfloat16(tensor):
        return np.dtype("uint16")
    if dt == np.dtype("float16"):
        return dt
    if dt == np.dtype("float32"):
        return dt
    if dt == np.dtype("float64"):
        return np.dtype("float32")
    return dt


def convert_to_storage(tensor: np.ndarray, storage_dtype: np.dtype) -> np.ndarray:
    if is_bfloat16(tensor):
        return tensor
    if storage_dtype == np.dtype("uint16") and not is_bfloat16(tensor):
        return float32_to_bfloat16(tensor.astype(np.float32))
    return tensor.astype(storage_dtype, copy=False)


def convert_from_storage(
    data: np.ndarray, storage_dtype: np.dtype, work_dtype=np.float32
) -> np.ndarray:
    if storage_dtype == np.dtype("uint16"):
        return bfloat16_to_float32(data)
    return data.astype(work_dtype, copy=False)


def encode_dtype_code(dt: np.dtype) -> int:
    return _DTYPE_CODE.get(dt, 0)


def decode_dtype_code(code: int) -> np.dtype:
    return _CODE_TO_DTYPE.get(code, np.dtype("float16"))

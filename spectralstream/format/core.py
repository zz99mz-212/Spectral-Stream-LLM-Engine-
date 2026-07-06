from __future__ import annotations

import hashlib
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SSF_MAGIC = b"SSF\x02"
SSF_HEADER_SIZE = 256
SSF_FOOTER_SIZE = 128
SSF_PAGE_SIZE = 4096
SSF_REDUNDANT_HEADER_OFFSET = 4096


class SSFVersion(IntEnum):
    V1 = 1
    V2 = 2
    V2_1 = 3


class TensorDType(IntEnum):
    F32 = 0
    F16 = 1
    BF16 = 2
    INT8 = 3
    INT4 = 4
    U8 = 5

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> TensorDType:
        m = {
            np.dtype("float32"): cls.F32,
            np.dtype("float64"): cls.F32,
            np.dtype("float16"): cls.F16,
            np.dtype("int8"): cls.INT8,
            np.dtype("uint8"): cls.U8,
            np.dtype("uint16"): cls.BF16,
        }
        try:
            m[np.dtype("bfloat16")] = cls.BF16
        except TypeError:
            pass
        result = m.get(np.dtype(dtype))
        if result is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return result

    def to_numpy(self) -> np.dtype:
        m = {
            self.F32: np.float32,
            self.F16: np.float16,
            self.INT8: np.int8,
            self.U8: np.uint8,
        }
        try:
            m[self.BF16] = np.dtype("bfloat16")
        except TypeError:
            m[self.BF16] = np.float16
        return m.get(self, np.float32)


_LOSSY_TO_METHOD = {
    0: 350,
    1: 350,
    2: 350,
    3: 352,
}
_LEGACY_COMPRESSION_MAP = {
    0: 0,
    1: 350,
    2: 350,
    3: 352,
}


def _align_up(val: int, align: int) -> int:
    return ((val + align - 1) // align) * align


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f}GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n}B"

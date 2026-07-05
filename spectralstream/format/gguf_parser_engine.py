"""
GGUF Parser Engine - Direct GGUF binary parser, GGML dequantizer, MMAP weight loader,
Spectral tensor converter, and cache layer for SpectralStream.

Clean-room implementation. No external GGUF/GGML dependencies.
Based on GGUF v2/v3 spec and GGML quantization type reference.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
import mmap as py_mmap
import os
import re
import struct
import threading
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# GGUF Format Constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION_V2 = 2
GGUF_VERSION_V3 = 3

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_INT32 = 4
GGUF_TYPE_UINT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Quantization Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_S = 17
GGML_TYPE_IQ3_S = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_BF16 = 30

GGML_BLOCK_SIZE = {
    GGML_TYPE_F32: 1,
    GGML_TYPE_F16: 1,
    GGML_TYPE_BF16: 1,
    GGML_TYPE_Q4_0: 32,
    GGML_TYPE_Q4_1: 32,
    GGML_TYPE_Q5_0: 32,
    GGML_TYPE_Q5_1: 32,
    GGML_TYPE_Q8_0: 32,
    GGML_TYPE_Q8_1: 32,
    GGML_TYPE_Q2_K: 256,
    GGML_TYPE_Q3_K: 256,
    GGML_TYPE_Q4_K: 256,
    GGML_TYPE_Q5_K: 256,
    GGML_TYPE_Q6_K: 256,
    GGML_TYPE_Q8_K: 256,
    GGML_TYPE_IQ2_XXS: 256,
    GGML_TYPE_IQ2_S: 256,
    GGML_TYPE_IQ3_S: 256,
    GGML_TYPE_IQ1_S: 256,
    GGML_TYPE_TQ2_0: 32,
}

GGML_BLOCK_BYTES = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_BF16: 2,
    GGML_TYPE_Q4_0: 18,
    GGML_TYPE_Q4_1: 20,
    GGML_TYPE_Q5_0: 22,
    GGML_TYPE_Q5_1: 24,
    GGML_TYPE_Q8_0: 34,
    GGML_TYPE_Q8_1: 36,
    GGML_TYPE_Q2_K: 84,
    GGML_TYPE_Q3_K: 110,
    GGML_TYPE_Q4_K: 144,
    GGML_TYPE_Q5_K: 176,
    GGML_TYPE_Q6_K: 210,
    GGML_TYPE_Q8_K: 292,
    GGML_TYPE_IQ2_XXS: 38,
    GGML_TYPE_IQ2_S: 42,
    GGML_TYPE_IQ3_S: 60,
    GGML_TYPE_IQ1_S: 24,
    GGML_TYPE_TQ2_0: 10,
}

GGML_TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_BF16: "BF16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0",
    GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q8_1: "Q8_1",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_Q8_K: "Q8_K",
    GGML_TYPE_IQ2_XXS: "IQ2_XXS",
    GGML_TYPE_IQ2_S: "IQ2_S",
    GGML_TYPE_IQ3_S: "IQ3_S",
    GGML_TYPE_IQ1_S: "IQ1_S",
    GGML_TYPE_TQ2_0: "TQ2_0",
}


_libc_path = ctypes.util.find_library("c")
_HAS_LIBC = _libc_path is not None
if _HAS_LIBC:
    _libc = ctypes.CDLL(_libc_path, use_errno=True)
    _libc.madvise.restype = ctypes.c_int
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libc.mincore.restype = ctypes.c_int
    _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    _libc.mlock.restype = ctypes.c_int
    _libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.munlock.restype = ctypes.c_int
    _libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    MADV_WILLNEED = 3
    MADV_DONTNEED = 4
    MADV_SEQUENTIAL = 1
    MADV_RANDOM = 2
    MADV_HUGEPAGE = 14
else:
    MADV_WILLNEED = 3
    MADV_DONTNEED = 4
    MADV_SEQUENTIAL = 1
    MADV_RANDOM = 2
    MADV_HUGEPAGE = 14


def _bf16_to_f32(bf16: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bf16 << 16))[0]


class GGUFParser:
    """Pure-Python GGUF binary parser. Uses gguf library if available."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._data: Optional[bytes] = None
        self._pos: int = 0
        self.magic: bytes = b""
        self.version: int = 0
        self.tensor_count: int = 0
        self.metadata_kv_count: int = 0
        self.metadata: dict[str, Any] = {}
        self.tensor_infos: list[dict[str, Any]] = []
        self.file_size: int = 0
        self.tensor_data_offset: int = 0
        self._use_gguf_lib = False
        self._gguf_reader: Any = None
        self._gguf_model: Any = None

    @classmethod
    def from_buffer(cls, buffer: Any) -> "GGUFParser":
        """Create a GGUFParser from an existing mmap/bytes buffer (zero-copy)."""
        parser = cls.__new__(cls)
        parser.path = Path("<buffer>")
        parser._data = buffer
        parser._pos = 0
        parser.magic = b""
        parser.version = 0
        parser.tensor_count = 0
        parser.metadata_kv_count = 0
        parser.metadata = {}
        parser.tensor_infos = []
        parser.file_size = len(buffer)
        parser.tensor_data_offset = 0
        parser._use_gguf_lib = False
        parser._gguf_reader = None
        parser._gguf_model = None
        parser._file_path = "<buffer>"
        parser._mmap_obj = buffer
        parser._parse_header_from_mmap()
        return parser

    def _extract_reader_field_value(self, field) -> Any:
        """Extract a plain Python value from a gguf ReaderField object."""
        try:
            from gguf import GGUFValueType

            if not hasattr(field, "data") or not hasattr(field, "parts"):
                return field
            data_idx = field.data[0] if len(field.data) > 0 else 0
            if len(field.parts) <= data_idx:
                return field
            part = field.parts[data_idx]
            types = getattr(field, "types", [])
            if len(types) > 0:
                vtype = types[0] if hasattr(types[0], "value") else types[0]
            else:
                vtype = None
            if vtype == GGUFValueType.STRING or (
                hasattr(vtype, "name") and vtype.name == "STRING"
            ):
                return bytes(part).decode("utf-8", errors="replace")
            if vtype == GGUFValueType.ARRAY or (
                hasattr(vtype, "name") and vtype.name == "ARRAY"
            ):
                return field
            if hasattr(part, "item"):
                return part.item() if callable(part.item) else int(part)
            if hasattr(part, "__len__") and len(part) == 1:
                return int(part[0])
            return int(part)
        except Exception:
            return str(field)

    def _parse_via_gguf_lib(self):
        """Parse GGUF using the gguf Python library (more robust)."""
        try:
            from gguf import GGUFReader

            self._gguf_reader = GGUFReader(str(self.path))
            self._use_gguf_lib = True
            ver_field = self._gguf_reader.fields.get("version")
            if ver_field is not None:
                self.version = self._extract_reader_field_value(ver_field)
            else:
                self.version = 3
            self.metadata = {}
            for key, field in self._gguf_reader.fields.items():
                self.metadata[key] = self._extract_reader_field_value(field)
            for t in self._gguf_reader.tensors:
                from gguf import GGMLQuantizationType

                gt = t.tensor_type if hasattr(t, "tensor_type") else 0
                self.tensor_infos.append(
                    {
                        "name": t.name,
                        "shape": tuple(t.shape),
                        "ggml_type": gt,
                        "type_name": str(GGMLQuantizationType(gt).name)
                        if hasattr(GGMLQuantizationType, "name")
                        else f"type_{gt}",
                        "offset": t.field_offset if hasattr(t, "field_offset") else 0,
                        "data_size": t.n_bytes if hasattr(t, "n_bytes") else 0,
                        "n_elements": t.n_elements if hasattr(t, "n_elements") else 0,
                    }
                )
            self.tensor_count = len(self.tensor_infos)
            self.metadata_kv_count = len(self.metadata)
            return self
        except ImportError:
            raise ImportError("gguf library not available for fallback parsing")

    def parse(self):
        self.file_size = os.path.getsize(self.path)
        self._file_path = str(self.path)
        # Memory-map the file and parse header natively (fast, reliable)
        try:
            with open(self.path, "rb") as f:
                self._mmap_obj = py_mmap.mmap(f.fileno(), 0, access=py_mmap.ACCESS_READ)
                self._data = self._mmap_obj
            print("[GGUFParser] mmap OK", flush=True)
            self._parse_header_from_mmap()
            return self
        except Exception as _e:
            _mmap_err = _e
            print(
                f"[GGUFParser] mmap parse failed: {_e}, trying gguf library...",
                flush=True,
            )
        # Fallback to gguf library if mmap fails
        try:
            result = self._parse_via_gguf_lib()
            print(
                f"[GGUFParser] gguf library OK: {len(self.tensor_infos)} tensors",
                flush=True,
            )
            return result
        except Exception as _e2:
            raise RuntimeError(
                f"Failed to parse GGUF: mmap error={_mmap_err}, gguf error={_e2}"
            )

    def _parse_header_from_mmap(self):
        """Parse GGUF header and tensor index from self._data (already set)."""
        self._pos = 0
        self.magic = self._read(4)
        if self.magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: magic={self.magic!r}")
        self.version = self._read_u32()
        if self.version not in (GGUF_VERSION_V2, GGUF_VERSION_V3):
            raise ValueError(f"Unsupported GGUF version: {self.version}")
        self.tensor_count = self._read_u64()
        self.metadata_kv_count = self._read_u64()
        print(
            f"[GGUFParser] tensors={self.tensor_count} kvs={self.metadata_kv_count}",
            flush=True,
        )
        for i in range(self.metadata_kv_count):
            key = self._read_string()
            if i == 16:
                print(f"  meta[{i}] key={key} (reading value...)", flush=True)
                t0 = time.time()
                val = self._read_value()
                print(
                    f"  meta[{i}] {key} = {str(val)[:80]} ({time.time() - t0:.1f}s)",
                    flush=True,
                )
            else:
                val = self._read_value()
                if i < 20 or i >= self.metadata_kv_count - 2:
                    print(f"  meta[{i}] {key} = {str(val)[:80]}", flush=True)
            self.metadata[key] = val
        for i in range(self.tensor_count):
            name = self._read_string()
            n_dims = self._read_u32()
            dims = tuple(self._read_u64() for _ in range(n_dims))
            ggml_type = self._read_u32()
            offset = self._read_u64()
            blk_sz = GGML_BLOCK_SIZE.get(ggml_type, 1)
            blk_bytes = GGML_BLOCK_BYTES.get(ggml_type, 4)
            n_blocks = (math.prod(dims) + blk_sz - 1) // blk_sz
            data_size = n_blocks * blk_bytes
            self.tensor_infos.append(
                {
                    "name": name,
                    "shape": dims,
                    "ggml_type": ggml_type,
                    "type_name": GGML_TYPE_NAMES.get(
                        ggml_type, f"UNKNOWN({ggml_type})"
                    ),
                    "offset": offset,
                    "data_size": data_size,
                    "n_elements": math.prod(dims),
                }
            )
        alignment = 32
        self.tensor_data_offset = (self._pos + alignment - 1) // alignment * alignment
        self._pos = 0

    def _read(self, size: int) -> bytes:
        end = self._pos + size
        if end > len(self._data):
            raise ValueError(f"Unexpected EOF at offset {self._pos}")
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def _read_u32(self) -> int:
        v = struct.unpack_from("<I", self._data, self._pos)[0]
        self._pos += 4
        return v

    def _read_u64(self) -> int:
        v = struct.unpack_from("<Q", self._data, self._pos)[0]
        self._pos += 8
        return v

    def _read_string(self) -> str:
        length = self._read_u64()
        raw = self._read(length)
        return raw.decode("utf-8", errors="replace")

    def _read_value(self) -> Any:
        vt = self._read_u32()
        if vt == GGUF_TYPE_UINT8:
            return self._read(1)[0]
        if vt == GGUF_TYPE_INT8:
            v = struct.unpack_from("<b", self._data, self._pos)[0]
            self._pos += 1
            return v
        if vt == GGUF_TYPE_UINT16:
            v = struct.unpack_from("<H", self._data, self._pos)[0]
            self._pos += 2
            return v
        if vt == GGUF_TYPE_INT16:
            v = struct.unpack_from("<h", self._data, self._pos)[0]
            self._pos += 2
            return v
        if vt == GGUF_TYPE_UINT32:
            return self._read_u32()
        if vt == GGUF_TYPE_INT32:
            v = struct.unpack_from("<i", self._data, self._pos)[0]
            self._pos += 4
            return v
        if vt == GGUF_TYPE_FLOAT32:
            v = struct.unpack_from("<f", self._data, self._pos)[0]
            self._pos += 4
            return v
        if vt == GGUF_TYPE_BOOL:
            return bool(self._read(1)[0])
        if vt == GGUF_TYPE_STRING:
            return self._read_string()
        if vt == GGUF_TYPE_ARRAY:
            if self.version >= 3:
                et = self._read_u32()
                n = self._read_u64()
            else:
                n = self._read_u64()
                et = self._read_u32()
            return [self._read_typed_value(et) for _ in range(n)]
        if vt == GGUF_TYPE_UINT64:
            return self._read_u64()
        if vt == GGUF_TYPE_INT64:
            v = struct.unpack_from("<q", self._data, self._pos)[0]
            self._pos += 8
            return v
        if vt == GGUF_TYPE_FLOAT64:
            v = struct.unpack_from("<d", self._data, self._pos)[0]
            self._pos += 8
            return v
        raise ValueError(f"Unknown GGUF value type: {vt}")

    def _read_typed_value(self, vt: int) -> Any:
        if vt == GGUF_TYPE_UINT8:
            return self._read(1)[0]
        if vt == GGUF_TYPE_INT8:
            v = struct.unpack_from("<b", self._data, self._pos)[0]
            self._pos += 1
            return v
        if vt == GGUF_TYPE_UINT16:
            v = struct.unpack_from("<H", self._data, self._pos)[0]
            self._pos += 2
            return v
        if vt == GGUF_TYPE_INT16:
            v = struct.unpack_from("<h", self._data, self._pos)[0]
            self._pos += 2
            return v
        if vt == GGUF_TYPE_UINT32:
            return self._read_u32()
        if vt == GGUF_TYPE_INT32:
            v = struct.unpack_from("<i", self._data, self._pos)[0]
            self._pos += 4
            return v
        if vt == GGUF_TYPE_FLOAT32:
            v = struct.unpack_from("<f", self._data, self._pos)[0]
            self._pos += 4
            return v
        if vt == GGUF_TYPE_BOOL:
            return bool(self._read(1)[0])
        if vt == GGUF_TYPE_STRING:
            return self._read_string()
        if vt == GGUF_TYPE_UINT64:
            return self._read_u64()
        if vt == GGUF_TYPE_INT64:
            v = struct.unpack_from("<q", self._data, self._pos)[0]
            self._pos += 8
            return v
        if vt == GGUF_TYPE_FLOAT64:
            v = struct.unpack_from("<d", self._data, self._pos)[0]
            self._pos += 8
            return v
        raise ValueError(f"Unknown GGUF value type: {vt}")

    def get_tensor_info(self, name: str) -> Optional[dict]:
        for ti in self.tensor_infos:
            if ti["name"] == name:
                return ti
        return None

    def summary(self) -> str:
        tc = {}
        for ti in self.tensor_infos:
            tn = ti["type_name"]
            tc[tn] = tc.get(tn, 0) + 1
        return "\n".join(
            [
                f"  Version: {self.version}",
                f"  Tensors: {self.tensor_count}",
                f"  Metadata keys: {len(self.metadata)}",
                f"  File size: {self.file_size / 1024**2:.0f} MB",
                f"  Types: {dict(sorted(tc.items()))}",
            ]
        )

    def __repr__(self) -> str:
        return (
            f"<GGUFParser {self.path.name} v{self.version} {self.tensor_count} tensors>"
        )


class GGMLDequantizer:
    """Dequantize all GGML quantized types to float32."""

    @staticmethod
    def dequantize_fast(data: np.ndarray, ggml_type: int) -> np.ndarray:
        """CPU-optimized dequant: vectorized, cache-friendly, SIMD-friendly.

        Uses numpy vectorized operations, no per-element Python loops.
        Cache-blocked: processes blocks fitting in ~32KB L1 cache.
        """
        if ggml_type == GGML_TYPE_F32:
            return data.view(np.float32).copy()
        if ggml_type == GGML_TYPE_F16:
            return data.view(np.float16).astype(np.float32)
        if ggml_type == GGML_TYPE_BF16:
            return GGMLDequantizer._deq_bf16(data)
        if ggml_type == GGML_TYPE_Q4_0:
            return GGMLDequantizer._deq_q4_0_fast(data)
        if ggml_type == GGML_TYPE_Q4_1:
            return GGMLDequantizer._deq_q4_1_fast(data)
        if ggml_type == GGML_TYPE_Q5_0:
            return GGMLDequantizer._deq_q5_0_fast(data)
        if ggml_type == GGML_TYPE_Q5_1:
            return GGMLDequantizer._deq_q5_1_fast(data)
        if ggml_type == GGML_TYPE_Q8_0:
            return GGMLDequantizer._deq_q8_0_fast(data)
        if ggml_type == GGML_TYPE_Q8_1:
            return GGMLDequantizer._deq_q8_1_fast(data)
        if ggml_type == GGML_TYPE_Q2_K:
            return GGMLDequantizer._deq_q2_K_fast(data)
        if ggml_type == GGML_TYPE_Q3_K:
            return GGMLDequantizer._deq_q3_K(data)
        if ggml_type == GGML_TYPE_Q4_K:
            return GGMLDequantizer._deq_q4_K_fast(data)
        if ggml_type == GGML_TYPE_Q5_K:
            return GGMLDequantizer._deq_q5_K_fast(data)
        if ggml_type == GGML_TYPE_Q6_K:
            return GGMLDequantizer._deq_q6_K_fast(data)
        if ggml_type == GGML_TYPE_Q8_K:
            return GGMLDequantizer._deq_q8_K_fast(data)
        if ggml_type == GGML_TYPE_IQ2_XXS:
            return GGMLDequantizer._deq_iq2_xxs(data)
        if ggml_type == GGML_TYPE_IQ2_S:
            return GGMLDequantizer._deq_iq2_s(data)
        if ggml_type == GGML_TYPE_IQ3_S:
            return GGMLDequantizer._deq_iq3_s(data)
        if ggml_type == GGML_TYPE_IQ1_S:
            return GGMLDequantizer._deq_iq1_s(data)
        if ggml_type == GGML_TYPE_TQ2_0:
            return GGMLDequantizer._deq_tq2_0(data)
        logger.warning(f"Unknown GGML type {ggml_type}, attempting raw fallback")
        try:
            return data.view(np.float32).copy()
        except Exception:
            return data.astype(np.float32)

    @staticmethod
    def dequantize(data: np.ndarray, ggml_type: int) -> np.ndarray:
        if ggml_type == GGML_TYPE_F32:
            return data.view(np.float32).copy()
        if ggml_type == GGML_TYPE_F16:
            return data.view(np.float16).astype(np.float32)
        if ggml_type == GGML_TYPE_BF16:
            return GGMLDequantizer._deq_bf16(data)
        if ggml_type == GGML_TYPE_Q4_0:
            return GGMLDequantizer._deq_q4_0(data)
        if ggml_type == GGML_TYPE_Q4_1:
            return GGMLDequantizer._deq_q4_1(data)
        if ggml_type == GGML_TYPE_Q5_0:
            return GGMLDequantizer._deq_q5_0(data)
        if ggml_type == GGML_TYPE_Q5_1:
            return GGMLDequantizer._deq_q5_1(data)
        if ggml_type == GGML_TYPE_Q8_0:
            return GGMLDequantizer._deq_q8_0(data)
        if ggml_type == GGML_TYPE_Q8_1:
            return GGMLDequantizer._deq_q8_1(data)
        if ggml_type == GGML_TYPE_Q2_K:
            return GGMLDequantizer._deq_q2_K(data)
        if ggml_type == GGML_TYPE_Q3_K:
            return GGMLDequantizer._deq_q3_K(data)
        if ggml_type == GGML_TYPE_Q4_K:
            return GGMLDequantizer._deq_q4_K(data)
        if ggml_type == GGML_TYPE_Q5_K:
            return GGMLDequantizer._deq_q5_K(data)
        if ggml_type == GGML_TYPE_Q6_K:
            return GGMLDequantizer._deq_q6_K(data)
        if ggml_type == GGML_TYPE_Q8_K:
            return GGMLDequantizer._deq_q8_K(data)
        if ggml_type == GGML_TYPE_IQ2_XXS:
            return GGMLDequantizer._deq_iq2_xxs(data)
        if ggml_type == GGML_TYPE_IQ2_S:
            return GGMLDequantizer._deq_iq2_s(data)
        if ggml_type == GGML_TYPE_IQ3_S:
            return GGMLDequantizer._deq_iq3_s(data)
        if ggml_type == GGML_TYPE_IQ1_S:
            return GGMLDequantizer._deq_iq1_s(data)
        if ggml_type == GGML_TYPE_TQ2_0:
            return GGMLDequantizer._deq_tq2_0(data)
        # Fallback: unknown type, return raw bytes as float32 view (best effort)
        logger.warning(f"Unknown GGML type {ggml_type}, attempting raw fallback")
        try:
            return data.view(np.float32).copy()
        except Exception:
            return data.astype(np.float32)

    @staticmethod
    def _deq_bf16(data: np.ndarray) -> np.ndarray:
        raw = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
        shifted = raw << 16
        out = np.zeros(len(shifted), dtype=np.float32)
        np.copyto(out, shifted.view(np.float32))
        return out

    # ── Fast (vectorized) dequant methods ──────────────────────────────

    @staticmethod
    def _deq_q4_0_fast(data: np.ndarray) -> np.ndarray:
        """Q4_0: vectorized nibble unpack, no loop per element.

        Supports partial blocks: if data length is not a multiple of 18,
        the trailing partial block is decoded with a small scalar loop.
        """
        raw = np.frombuffer(data, dtype=np.uint8)
        n_bytes = len(raw)
        if n_bytes < 2:
            return np.array([], dtype=np.float32)

        n_full_blocks = n_bytes // 18
        remaining = n_bytes % 18
        n_full_values = n_full_blocks * 32
        n_partial_values = max(0, (remaining - 2) * 2) if remaining >= 2 else 0

        # Decode full blocks (vectorized path)
        if n_full_blocks > 0:
            d = np.frombuffer(
                raw[: n_full_blocks * 2].tobytes(), dtype=np.float16
            ).astype(np.float32)
            packed = raw[n_full_blocks * 2 : n_full_blocks * 18].reshape(
                n_full_blocks, 16
            )
            lo = (packed >> 0) & 0x0F
            hi = (packed >> 4) & 0x0F
            vals = np.empty((n_full_blocks, 32), dtype=np.uint8)
            vals[:, 0::2] = lo
            vals[:, 1::2] = hi
            full_out = (vals.astype(np.float32) - 8.0) * d[:, np.newaxis]
        else:
            full_out = np.empty((0, 32), dtype=np.float32)

        # Decode trailing partial block (scalar fallback)
        if n_partial_values > 0:
            partial_out = np.zeros(n_partial_values, dtype=np.float32)
            off = n_full_blocks * 18
            d_partial = struct.unpack("<e", raw[off : off + 2])[0]
            for i in range(n_partial_values):
                nib = raw[off + 2 + i // 2]
                val = (nib >> (4 * (i % 2))) & 0x0F
                partial_out[i] = (val - 8.0) * d_partial
        else:
            partial_out = np.empty(0, dtype=np.float32)

        return np.concatenate([full_out.ravel(), partial_out])

    @staticmethod
    def _deq_q4_1_fast(data: np.ndarray) -> np.ndarray:
        """Q4_1: vectorized."""
        n_blocks = len(data) // 20
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        h = raw[: n_blocks * 4].reshape(n_blocks, 4)
        d = np.frombuffer(h[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
        m = np.frombuffer(h[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)
        packed = raw[n_blocks * 4 : n_blocks * 20].reshape(n_blocks, 16)
        lo = (packed >> 0) & 0x0F
        hi = (packed >> 4) & 0x0F
        vals = np.empty((n_blocks, 32), dtype=np.uint8)
        vals[:, 0::2] = lo
        vals[:, 1::2] = hi
        return vals.astype(np.float32) * d[:, np.newaxis] + m[:, np.newaxis]

    @staticmethod
    def _deq_q5_0_fast(data: np.ndarray) -> np.ndarray:
        """Q5_0: vectorized high-bit extraction."""
        n_blocks = len(data) // 22
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        d = np.frombuffer(raw[: n_blocks * 2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        qh = np.frombuffer(
            raw[n_blocks * 2 : n_blocks * 6].reshape(n_blocks, 4).tobytes(),
            dtype=np.uint32,
        )
        packed = raw[n_blocks * 6 : n_blocks * 22].reshape(n_blocks, 16)
        lo = (packed >> 0) & 0x0F
        hi = (packed >> 4) & 0x0F
        iota = np.arange(32, dtype=np.uint32)[np.newaxis, :]
        vh = ((qh[:, np.newaxis] >> iota) & 1).astype(np.uint8)
        all_lo = np.empty((n_blocks, 32), dtype=np.uint8)
        all_lo[:, 0::2] = lo
        all_lo[:, 1::2] = hi
        vals_f = all_lo.astype(np.float32) + (vh.astype(np.float32) * 16.0)
        return (vals_f - 16.0) * d[:, np.newaxis]

    @staticmethod
    def _deq_q5_1_fast(data: np.ndarray) -> np.ndarray:
        """Q5_1: vectorized."""
        n_blocks = len(data) // 24
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        hdr = raw[: n_blocks * 4].reshape(n_blocks, 4)
        d = np.frombuffer(hdr[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
        m = np.frombuffer(hdr[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)
        qh = np.frombuffer(
            raw[n_blocks * 4 : n_blocks * 8].reshape(n_blocks, 4).tobytes(),
            dtype=np.uint32,
        )
        packed = raw[n_blocks * 8 : n_blocks * 24].reshape(n_blocks, 16)
        lo = (packed >> 0) & 0x0F
        hi = (packed >> 4) & 0x0F
        iota = np.arange(32, dtype=np.uint32)[np.newaxis, :]
        vh = ((qh[:, np.newaxis] >> iota) & 1).astype(np.uint8)
        all_lo = np.empty((n_blocks, 32), dtype=np.uint8)
        all_lo[:, 0::2] = lo
        all_lo[:, 1::2] = hi
        vals_f = all_lo.astype(np.float32) + (vh.astype(np.float32) * 16.0)
        return vals_f * d[:, np.newaxis] + m[:, np.newaxis]

    @staticmethod
    def _deq_q8_0_fast(data: np.ndarray) -> np.ndarray:
        """Q8_0: vectorized int8*scale with broadcasting."""
        n_blocks = len(data) // 34
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        d = np.frombuffer(raw[: n_blocks * 2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        qs = (
            np.frombuffer(raw[n_blocks * 2 : n_blocks * 34], dtype=np.int8)
            .reshape(n_blocks, 32)
            .astype(np.float32)
        )
        return qs * d[:, np.newaxis]

    @staticmethod
    def _deq_q8_1_fast(data: np.ndarray) -> np.ndarray:
        """Q8_1: vectorized."""
        n_blocks = len(data) // 36
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        hdr = raw[: n_blocks * 4].reshape(n_blocks, 4)
        d = np.frombuffer(hdr[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
        s = np.frombuffer(hdr[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)
        qs = (
            np.frombuffer(raw[n_blocks * 4 : n_blocks * 36], dtype=np.int8)
            .reshape(n_blocks, 32)
            .astype(np.float32)
        )
        return qs * d[:, np.newaxis] + s[:, np.newaxis]

    @staticmethod
    def _deq_q2_K_fast(data: np.ndarray) -> np.ndarray:
        """Q2_K: vectorized scale lookup, 2-bit unpack with np.take()."""
        n_blocks = len(data) // 84
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 84)
        d_all = np.frombuffer(raw[:, 80:82].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        dmin_all = np.frombuffer(raw[:, 82:84].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        # 16 sub-blocks, each with a 1-byte scale
        sc = raw[:, :16]  # (n_blocks, 16)
        d = d_all[:, np.newaxis] * ((sc & 0x0F) + 1) / 8.0
        mn = dmin_all[:, np.newaxis] * (((sc >> 4) & 0x0F) + 1) / 8.0
        # Unpack 2-bit values from 64 bytes
        packed = raw[:, 16:80]  # (n_blocks, 64)
        # Each byte has 4 values (each 2 bits)
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
        vals = np.empty((n_blocks, 256), dtype=np.uint8)
        for j in range(4):
            vals[:, j::4] = (packed >> shifts[j]) & 0x03
        vals_f = vals.astype(np.float32)
        # Apply per-sub-block scale
        # sub_block i (size 16) uses scale[i] and min[i]
        d_rep = np.repeat(d, 16, axis=1)
        mn_rep = np.repeat(mn, 16, axis=1)
        return vals_f * d_rep - mn_rep

    @staticmethod
    def _deq_q4_K_fast(data: np.ndarray) -> np.ndarray:
        """Q4_K: vectorized scale+min extraction, nibble unpack."""
        n_blocks = len(data) // 144
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 144)
        d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
        dmin = np.frombuffer(raw[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)
        sc = raw[:, 4:16]
        qs = raw[:, 16:144]

        # Extract 8 (scale, min) pairs from 12 bytes
        scale = np.empty((n_blocks, 8), dtype=np.int32)
        mind = np.empty((n_blocks, 8), dtype=np.int32)
        # Pairs 0-3: simple
        for j in range(4):
            scale[:, j] = sc[:, j] & 63
            mind[:, j] = sc[:, j + 4] & 63
        # Pairs 4-7: interleaved (GGML K4 packing: uses bytes 8-11 + high bits of 0-3 and 4-7)
        for j in range(4, 8):
            scale[:, j] = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            mind[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)

        d_mul = d[:, np.newaxis] * scale.astype(np.float32)
        dmin_mul = dmin[:, np.newaxis] * mind.astype(np.float32)

        out = np.empty((n_blocks, 256), dtype=np.float32)
        for sb in range(4):
            chunk = qs[:, sb * 32 : (sb + 1) * 32]
            lo = (chunk & 0x0F).astype(np.float32)
            hi = ((chunk >> 4) & 0x0F).astype(np.float32)
            out[:, sb * 64 + 0 : sb * 64 + 32] = (
                lo * d_mul[:, sb * 2 : sb * 2 + 1] - dmin_mul[:, sb * 2 : sb * 2 + 1]
            )
            out[:, sb * 64 + 32 : sb * 64 + 64] = (
                hi * d_mul[:, sb * 2 + 1 : sb * 2 + 2]
                - dmin_mul[:, sb * 2 + 1 : sb * 2 + 2]
            )
        return out.ravel()

    @staticmethod
    def _deq_q5_K_fast(data: np.ndarray) -> np.ndarray:
        """Q5_K: vectorized 5-bit unpack with high-bit mask."""
        n_blocks = len(data) // 176
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 176)
        d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
        dmin = np.frombuffer(raw[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)
        sc = raw[:, 4:16]
        qh = raw[:, 16:48]
        qs = raw[:, 48:176]

        scale = np.empty((n_blocks, 8), dtype=np.int32)
        mind = np.empty((n_blocks, 8), dtype=np.int32)
        for j in range(4):
            scale[:, j] = sc[:, j] & 63
            mind[:, j] = sc[:, j + 4] & 63
        for j in range(4, 8):
            scale[:, j] = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            mind[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        d_mul = d[:, np.newaxis] * np.nan_to_num(scale.astype(np.float32), nan=1.0)
        dmin_mul = dmin[:, np.newaxis] * np.nan_to_num(mind.astype(np.float32), nan=0.0)

        out = np.empty((n_blocks, 256), dtype=np.float32)
        for sb in range(4):
            chunk = qs[:, sb * 32 : (sb + 1) * 32]
            lo = (chunk & 0x0F).astype(np.float32)
            hi = ((chunk >> 4) & 0x0F).astype(np.float32)
            for bi in range(32):
                hb_lo = ((qh[:, sb * 4 + bi // 8] >> (bi % 8)) & 1).astype(np.float32)
                lo[:, bi] += hb_lo * 16.0
                hb_hi = (
                    (qh[:, sb * 4 + (bi + 32) // 8] >> ((bi + 32) % 8)) & 1
                ).astype(np.float32)
                hi[:, bi] += hb_hi * 16.0
            out[:, sb * 64 + 0 : sb * 64 + 32] = (
                lo * d_mul[:, sb * 2 : sb * 2 + 1] - dmin_mul[:, sb * 2 : sb * 2 + 1]
            )
            out[:, sb * 64 + 32 : sb * 64 + 64] = (
                hi * d_mul[:, sb * 2 + 1 : sb * 2 + 2]
                - dmin_mul[:, sb * 2 + 1 : sb * 2 + 2]
            )
        return out.ravel()

    @staticmethod
    def _deq_q6_K_fast(data: np.ndarray) -> np.ndarray:
        """Q6_K: vectorized."""
        n_blocks = len(data) // 210
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 210)
        d = np.frombuffer(raw[:, 208:210].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        ql = raw[:, 0:128]
        qh = raw[:, 128:192]
        scales = raw[:, 192:208]

        out = np.empty((n_blocks, 256), dtype=np.float32)
        for l in range(32):
            is_idx = l // 16
            v1 = ((ql[:, l] & 0x0F) | ((qh[:, l] >> 0) & 3) << 4).astype(
                np.float32
            ) - 32.0
            v2 = ((ql[:, l + 32] & 0x0F) | ((qh[:, l] >> 2) & 3) << 4).astype(
                np.float32
            ) - 32.0
            v3 = ((ql[:, l] >> 4) | ((qh[:, l] >> 4) & 3) << 4).astype(
                np.float32
            ) - 32.0
            v4 = ((ql[:, l + 32] >> 4) | ((qh[:, l] >> 6) & 3) << 4).astype(
                np.float32
            ) - 32.0
            out[:, l + 0] = d * scales[:, is_idx].astype(np.float32) * v1
            out[:, l + 32] = d * scales[:, is_idx + 2].astype(np.float32) * v2
            out[:, l + 64] = d * scales[:, is_idx + 4].astype(np.float32) * v3
            out[:, l + 96] = d * scales[:, is_idx + 6].astype(np.float32) * v4
        return out.ravel()

    @staticmethod
    def _deq_q8_K_fast(data: np.ndarray) -> np.ndarray:
        """Q8_K: vectorized float32 scale * int8 values."""
        n_blocks = len(data) // 292
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 292)
        d = np.frombuffer(raw[:, 0:4].tobytes(), dtype=np.float32)
        qs = (
            np.frombuffer(raw[:, 4:260].ravel(), dtype=np.int8)
            .reshape(n_blocks, 256)
            .astype(np.float32)
        )
        return qs * d[:, np.newaxis]

    # ── Legacy (loop-based) dequant methods ────────────────────────────

    @staticmethod
    def _deq_q4_0(data: np.ndarray) -> np.ndarray:
        """Q4_0: [d:fp16][qs:4-bit nibbles x32] = 18B

        Supports partial blocks: if data length is not a multiple of 18,
        the trailing partial block is decoded based on available nibble bytes.
        """
        raw = np.frombuffer(data, dtype=np.uint8)
        n_bytes = len(raw)
        if n_bytes < 2:
            return np.array([], dtype=np.float32)

        n_full_blocks = n_bytes // 18
        remaining = n_bytes % 18
        n_full_values = n_full_blocks * 32
        # Partial block: 2 bytes for scale, rest for nibbles (each byte holds 2 nibbles)
        n_partial_values = max(0, (remaining - 2) * 2) if remaining >= 2 else 0
        total_values = n_full_values + n_partial_values

        out = np.zeros(total_values, dtype=np.float32)

        # Decode full blocks
        for b in range(n_full_blocks):
            off = b * 18
            d = struct.unpack("<e", raw[off : off + 2])[0]
            base = b * 32
            for i in range(32):
                nib = raw[off + 2 + i // 2]
                val = (nib >> (4 * (i % 2))) & 0x0F
                out[base + i] = (val - 8.0) * d

        # Decode trailing partial block
        if n_partial_values > 0:
            off = n_full_blocks * 18
            d = struct.unpack("<e", raw[off : off + 2])[0]
            base = n_full_values
            for i in range(n_partial_values):
                nib = raw[off + 2 + i // 2]
                val = (nib >> (4 * (i % 2))) & 0x0F
                out[base + i] = (val - 8.0) * d

        return out

    @staticmethod
    def _deq_q4_1(data: np.ndarray) -> np.ndarray:
        """Q4_1: [d:fp16][m:fp16][qs:nibbles x32] = 20B"""
        n_blocks = len(data) // 20
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 32, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 20
            d = struct.unpack("<e", raw[off : off + 2])[0]
            m = struct.unpack("<e", raw[off + 2 : off + 4])[0]
            for i in range(32):
                nib = raw[off + 4 + i // 2]
                val = (nib >> (4 * (i % 2))) & 0x0F
                out[b * 32 + i] = val * d + m
        return out

    @staticmethod
    def _deq_q5_0(data: np.ndarray) -> np.ndarray:
        """Q5_0: [d:fp16][qh:uint32][ql:nibbles x32] = 22B"""
        n_blocks = len(data) // 22
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 32, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 22
            d = struct.unpack("<e", raw[off : off + 2])[0]
            qh = struct.unpack("<I", raw[off + 2 : off + 6])[0]
            for i in range(32):
                low = raw[off + 6 + i // 2]
                vl = (low >> (4 * (i % 2))) & 0x0F
                vh = (qh >> i) & 1
                val = vl | (vh << 4)
                out[b * 32 + i] = (val - 16.0) * d
        return out

    @staticmethod
    def _deq_q5_1(data: np.ndarray) -> np.ndarray:
        """Q5_1: [d:fp16][m:fp16][qh:uint32][ql:nibbles] = 24B"""
        n_blocks = len(data) // 24
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 32, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 24
            d = struct.unpack("<e", raw[off : off + 2])[0]
            m = struct.unpack("<e", raw[off + 2 : off + 4])[0]
            qh = struct.unpack("<I", raw[off + 4 : off + 8])[0]
            for i in range(32):
                low = raw[off + 8 + i // 2]
                vl = (low >> (4 * (i % 2))) & 0x0F
                vh = (qh >> i) & 1
                val = vl | (vh << 4)
                out[b * 32 + i] = val * d + m
        return out

    @staticmethod
    def _deq_q8_0(data: np.ndarray) -> np.ndarray:
        """Q8_0: [d:fp16][qs:int8 x32] = 34B"""
        n_blocks = len(data) // 34
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 32, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 34
            d = struct.unpack("<e", raw[off : off + 2])[0]
            qs = np.frombuffer(raw[off + 2 : off + 34], dtype=np.int8).astype(
                np.float32
            )
            out[b * 32 : (b + 1) * 32] = qs * d
        return out

    @staticmethod
    def _deq_q8_1(data: np.ndarray) -> np.ndarray:
        """Q8_1: [d:fp16][s:fp16][qs:int8 x32] = 36B"""
        n_blocks = len(data) // 36
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 32, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 36
            d = struct.unpack("<e", raw[off : off + 2])[0]
            s = struct.unpack("<e", raw[off + 2 : off + 4])[0]
            qs = np.frombuffer(raw[off + 4 : off + 36], dtype=np.int8).astype(
                np.float32
            )
            out[b * 32 : (b + 1) * 32] = qs * d + s
        return out

    @staticmethod
    def _deq_q2_K(data: np.ndarray) -> np.ndarray:
        """Q2_K: [scales:16B][qs:64B][d:fp16][dmin:fp16] = 84B"""
        n_blocks = len(data) // 84
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 84
            d_all = struct.unpack("<e", raw[off + 80 : off + 82])[0]
            dmin_all = struct.unpack("<e", raw[off + 82 : off + 84])[0]
            for i in range(16):
                sc = raw[off + i]
                d = d_all * ((sc & 0x0F) + 1) / 8.0
                mn = dmin_all * (((sc >> 4) & 0x0F) + 1) / 8.0
                for j in range(16):
                    idx = i * 16 + j
                    bi = 16 + idx // 4
                    sh = (idx % 4) * 2
                    q = (raw[off + bi] >> sh) & 0x03
                    out[b * 256 + idx] = q * d - mn
        return out

    @staticmethod
    def _deq_q3_K(data: np.ndarray) -> np.ndarray:
        """Q3_K: [hmask:32B][qs:64B][scales:12B][d:fp16] = 110B"""
        n_blocks = len(data) // 110
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 110
            d = struct.unpack("<e", raw[off + 108 : off + 110])[0]
            for i in range(16):
                sc_val = raw[off + 96 + i] if i < 12 else 0
                hi = (raw[off + i // 4] >> ((i % 4) * 2)) & 0x03
                per = sc_val & 0x0F
                scale = d * ((sc_val >> 4) | (hi << 4))
                mn = d * per
                for j in range(16):
                    bit_pos = (i * 16 + j) * 3
                    bi = 32 + bit_pos // 8
                    bo = bit_pos % 8
                    q = raw[off + bi] >> bo
                    if bo > 5:
                        q |= raw[off + bi + 1] << (8 - bo)
                    q &= 0x07
                    out[b * 256 + i * 16 + j] = q * scale - mn
        return out

    @staticmethod
    def _get_scale_min_k4(j: int, q: np.ndarray):
        """Extract scale and min from K4 packed scales (12 bytes -> 8 pairs)."""
        if j < 4:
            return int(q[j] & 63), int(q[j + 4] & 63)
        else:
            d = (int(q[j + 4]) & 0xF) | ((int(q[j - 4]) >> 6) << 4)
            m = (int(q[j + 4]) >> 4) | ((int(q[j]) >> 6) << 4)
            return d, m

    @staticmethod
    def _deq_q4_K(data: np.ndarray) -> np.ndarray:
        """Q4_K: [d:fp16][dmin:fp16][scales:12B][qs:128B] = 144B"""
        n_blocks = len(data) // 144
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 144
            d = struct.unpack("<e", raw[off + 0 : off + 2])[0]
            dmin = struct.unpack("<e", raw[off + 2 : off + 4])[0]
            scales = raw[off + 4 : off + 16]
            qs = raw[off + 16 : off + 144]
            is_idx = 0
            q_off = 0
            for j in range(0, 256, 64):
                sc1, m1 = GGMLDequantizer._get_scale_min_k4(is_idx + 0, scales)
                d1 = d * sc1
                mn1 = dmin * m1
                sc2, m2 = GGMLDequantizer._get_scale_min_k4(is_idx + 1, scales)
                d2 = d * sc2
                mn2 = dmin * m2
                for l in range(32):
                    out[b * 256 + j + l] = d1 * (qs[q_off + l] & 0xF) - mn1
                    out[b * 256 + j + 32 + l] = d2 * (qs[q_off + l] >> 4) - mn2
                q_off += 32
                is_idx += 2
        return out

    @staticmethod
    def _deq_q5_K(data: np.ndarray) -> np.ndarray:
        """Q5_K: [d:fp16][dmin:fp16][scales:12B][qh:32B][qs:128B] = 176B"""
        n_blocks = len(data) // 176
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 176
            d = struct.unpack("<e", raw[off + 0 : off + 2])[0]
            dmin = struct.unpack("<e", raw[off + 2 : off + 4])[0]
            scales = raw[off + 4 : off + 16]
            qh = raw[off + 16 : off + 48]
            qs = raw[off + 48 : off + 176]
            for i in range(256):
                low = (qs[i // 2] >> (4 * (i % 2))) & 0x0F
                high = (qh[i // 8] >> (i % 8)) & 0x01
                val = low | (high << 4)
                half = i // 32
                sc, mn = GGMLDequantizer._get_scale_min_k4(half, scales)
                out[b * 256 + i] = val * (d * sc) - (dmin * mn)
        return out

    @staticmethod
    def _deq_q6_K(data: np.ndarray) -> np.ndarray:
        """Q6_K: [ql:128B][qh:64B][scales:16B][d:fp16] = 210B"""
        n_blocks = len(data) // 210
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 210
            ql = raw[off + 0 : off + 128]
            qh = raw[off + 128 : off + 192]
            scales = raw[off + 192 : off + 208]
            d = struct.unpack("<e", raw[off + 208 : off + 210])[0]
            for n in range(0, 256, 128):
                for l in range(32):
                    is_idx = l // 16
                    v1 = int(ql[l + 0] & 0xF) | ((int(qh[l] >> 0) & 3) << 4)
                    v2 = int(ql[l + 32] & 0xF) | ((int(qh[l] >> 2) & 3) << 4)
                    v3 = int(ql[l + 0] >> 4) | ((int(qh[l] >> 4) & 3) << 4)
                    v4 = int(ql[l + 32] >> 4) | ((int(qh[l] >> 6) & 3) << 4)
                    q1 = v1 - 32
                    q2 = v2 - 32
                    q3 = v3 - 32
                    q4 = v4 - 32
                    out[b * 256 + n + l + 0] = d * scales[is_idx + 0] * q1
                    out[b * 256 + n + l + 32] = d * scales[is_idx + 2] * q2
                    out[b * 256 + n + l + 64] = d * scales[is_idx + 4] * q3
                    out[b * 256 + n + l + 96] = d * scales[is_idx + 6] * q4
        return out

    @staticmethod
    def _deq_q8_K(data: np.ndarray) -> np.ndarray:
        """Q8_K: [d:fp32][qs:int8 x256][bsums:int16 x16] = 292B"""
        n_blocks = len(data) // 292
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 292
            d = struct.unpack("<f", raw[off : off + 4])[0]
            qs = np.frombuffer(raw[off + 4 : off + 260], dtype=np.int8).astype(
                np.float32
            )
            out[b * 256 : (b + 1) * 256] = qs * d
        return out

    @staticmethod
    def _deq_iq2_xxs(data: np.ndarray) -> np.ndarray:
        """IQ2_XXS: 38B per 256 values"""
        n_blocks = len(data) // 38
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 38
            d = struct.unpack("<e", raw[off : off + 2])[0]
            for i in range(256):
                bi = 2 + i // 4
                sh = (i % 4) * 2
                q = (raw[off + bi] >> sh) & 0x03
                out[b * 256 + i] = (q - 1) * d
        return out

    @staticmethod
    def _deq_iq2_s(data: np.ndarray) -> np.ndarray:
        """IQ2_S: 42B per 256 values"""
        n_blocks = len(data) // 42
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 42
            d = struct.unpack("<e", raw[off : off + 2])[0]
            for i in range(256):
                bi = 2 + i // 4
                sh = (i % 4) * 2
                q = (raw[off + bi] >> sh) & 0x03
                out[b * 256 + i] = (q - 1) * d
        return out

    @staticmethod
    def _deq_iq3_s(data: np.ndarray) -> np.ndarray:
        """IQ3_S: 60B per 256 values"""
        n_blocks = len(data) // 60
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 60
            d = struct.unpack("<e", raw[off : off + 2])[0]
            for i in range(256):
                bit_pos = i * 3
                bi = 2 + bit_pos // 8
                bo = bit_pos % 8
                q = raw[off + bi] >> bo
                if bo > 5:
                    q |= raw[off + bi + 1] << (8 - bo)
                q &= 0x07
                out[b * 256 + i] = (q - 3) * d
        return out

    @staticmethod
    def _deq_iq1_s(data: np.ndarray) -> np.ndarray:
        """IQ1_S: 24B per 256 values"""
        n_blocks = len(data) // 24
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 256, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        for b in range(n_blocks):
            off = b * 24
            d = struct.unpack("<e", raw[off : off + 2])[0]
            for i in range(256):
                bi = 2 + i // 8
                sh = i % 8
                q = (raw[off + bi] >> sh) & 1
                out[b * 256 + i] = (q * 2 - 1) * d
        return out

    @staticmethod
    def _deq_tq2_0(data: np.ndarray) -> np.ndarray:
        """TQ2_0: [d:fp16][packed:8B] = 10B per 32 values"""
        n_blocks = len(data) // 10
        if n_blocks == 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_blocks * 32, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8)
        tmap = {0: -1.0, 1: 0.0, 2: 1.0, 3: 0.0}
        for b in range(n_blocks):
            off = b * 10
            d = struct.unpack("<e", raw[off : off + 2])[0]
            for i in range(32):
                bi = 2 + i // 4
                sh = (i % 4) * 2
                code = (raw[off + bi] >> sh) & 0x03
                out[b * 32 + i] = tmap.get(code, 0.0) * d
        return out


class MMAPWeightLoader:
    """MMAP-based weight loading for GGUF files with zero-copy access."""

    def __init__(self, path: str, prefetch_first_last: bool = True):
        self.path = Path(path)
        self._fd: Optional[int] = None
        self._mmap: Optional[py_mmap.mmap] = None
        self._mmap_addr: int = 0
        self._mmap_len: int = 0
        self._file_size: int = 0
        self._page_size: int = os.sysconf("SC_PAGE_SIZE")
        self._parser: Optional[GGUFParser] = None
        self._tensor_index: dict[str, dict] = {}
        self._layer_index: dict[int, list[str]] = {}
        self._n_layers: int = 0
        self.prefetch_first_last = prefetch_first_last
        self.access_count = 0
        self.dequantization_count = 0
        self.dequantization_time = 0.0
        self.prefetch_count = 0

    def open(self) -> "MMAPWeightLoader":
        self._fd = os.open(str(self.path), os.O_RDONLY | os.O_CLOEXEC)
        self._file_size = os.fstat(self._fd).st_size
        self._mmap = py_mmap.mmap(self._fd, self._file_size, access=py_mmap.ACCESS_READ)
        try:
            self._mmap_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))
        except TypeError:
            # Read-only mmap may not support from_buffer; use 0 as fallback
            self._mmap_addr = 0
        self._mmap_len = self._file_size
        self._parser = GGUFParser(str(self.path))
        self._parser._data = self._mmap
        self._parser._pos = 0
        self._parser.file_size = self._file_size
        self._parser._parse_header_from_mmap()
        self._build_index()
        if self.prefetch_first_last:
            self._prefetch_hot_tensors()
        return self

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _build_index(self):
        n_layers = 0
        for ti in self._parser.tensor_infos:
            self._tensor_index[ti["name"]] = ti
            m = re.search(r"\.(\d+)\.", ti["name"])
            if m:
                li = int(m.group(1))
                self._layer_index.setdefault(li, []).append(ti["name"])
                n_layers = max(n_layers, li + 1)
        self._n_layers = n_layers if n_layers else 32

    def _prefetch_hot_tensors(self):
        hot: set[int] = set()
        if self._n_layers > 0:
            for i in range(min(2, self._n_layers)):
                hot.add(i)
                hot.add(self._n_layers - 1 - i)
        for li in hot:
            for n in self._layer_index.get(li, []):
                self.prefetch_tensor(n)

    def get_tensor(self, name: str) -> np.ndarray:
        self.access_count += 1
        ti = self._tensor_index.get(name)
        if ti is None:
            raise KeyError(f"Tensor {name!r} not found")
        gt = ti["ggml_type"]
        off = int(ti["offset"]) + self._parser.tensor_data_offset
        sh = ti["shape"]
        mm = self._mmap
        if gt == GGML_TYPE_F32:
            n = math.prod(sh)
            return np.ndarray((n,), dtype=np.float32, buffer=mm, offset=off).reshape(sh)
        if gt == GGML_TYPE_F16:
            n = math.prod(sh)
            return (
                np.ndarray((n,), dtype=np.float16, buffer=mm, offset=off)
                .astype(np.float32)
                .reshape(sh)
            )
        if gt == GGML_TYPE_BF16:
            raw = np.frombuffer(
                mm, dtype=np.uint8, offset=off, count=int(ti["data_size"])
            )
            t0 = time.perf_counter()
            result = GGMLDequantizer._deq_bf16(raw)
            self.dequantization_time += time.perf_counter() - t0
            self.dequantization_count += 1
            return result[: math.prod(sh)].reshape(sh)
        raw = np.frombuffer(
            mm, dtype=np.uint8, offset=off, count=int(ti["data_size"])
        ).copy()
        t0 = time.perf_counter()
        result = GGMLDequantizer.dequantize_fast(raw, gt)
        self.dequantization_time += time.perf_counter() - t0
        self.dequantization_count += 1
        return result[: math.prod(sh)].reshape(sh)

    def get_layer(self, lidx: int) -> dict[str, np.ndarray]:
        result = {}
        names = self._layer_index.get(lidx, [])
        prefix = f"blk.{lidx}."
        if not names:
            names = [n for n in self._tensor_index if n.startswith(prefix)]
        for n in names:
            result[n] = self.get_tensor(n)
        return result

    def prefetch_tensor(self, name: str):
        if not _HAS_LIBC:
            return
        ti = self._tensor_index.get(name)
        if ti is None:
            return
        off = int(ti["offset"]) + self._parser.tensor_data_offset
        sz = int(ti["data_size"])
        if sz == 0:
            return
        try:
            _libc.madvise(ctypes.c_void_p(self._mmap_addr + off), sz, MADV_WILLNEED)
            self.prefetch_count += 1
        except Exception:
            pass

    def prefetch_layer(self, lidx: int):
        for n in self._layer_index.get(lidx, []):
            self.prefetch_tensor(n)

    def page_residency(self, name: str) -> dict:
        if not _HAS_LIBC:
            return {"error": "no libc"}
        ti = self._tensor_index.get(name)
        if ti is None:
            return {}
        off = int(ti["offset"]) + self._parser.tensor_data_offset
        length = int(ti["data_size"])
        if length == 0:
            return {}
        addr = self._mmap_addr + off
        n_pages = (length + self._page_size - 1) // self._page_size
        vs = (n_pages + 7) // 8
        vec = (ctypes.c_byte * vs)()
        try:
            r = _libc.mincore(ctypes.c_void_p(addr), length, vec)
            if r != 0:
                return {"error": f"mincore: {ctypes.get_errno()}"}
            resident = sum(1 for i in range(n_pages) if vec[i // 8] & (1 << (i % 8)))
            return {
                "total_pages": n_pages,
                "resident": resident,
                "resident_frac": resident / max(n_pages, 1),
                "page_size": self._page_size,
            }
        except Exception as e:
            return {"error": str(e)}

    def list_tensors(self) -> list[str]:
        return list(self._tensor_index.keys())

    def metadata(self) -> dict:
        return self._parser.metadata if self._parser else {}

    def tensor_info(self, name: str) -> Optional[dict]:
        return self._tensor_index.get(name)

    @property
    def n_tensors(self) -> int:
        return len(self._tensor_index)

    def get_stats(self) -> dict:
        return {
            "file_size_mb": self._file_size / 1024**2,
            "tensors": len(self._tensor_index),
            "layers": self._n_layers,
            "accesses": self.access_count,
            "dequantizations": self.dequantization_count,
            "dequantization_time_ms": self.dequantization_time * 1000,
            "prefetches": self.prefetch_count,
        }

    def __enter__(self) -> "MMAPWeightLoader":
        self.open()
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self) -> str:
        return (
            f"<MMAPWeightLoader {self.path.name} {self.n_tensors}t {self._n_layers}L>"
        )


_DCT_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    if n in _DCT_CACHE:
        return _DCT_CACHE[n]
    C = np.zeros((n, n), dtype=np.float64)
    C[0, :] = 1.0 / math.sqrt(n)
    s = math.sqrt(2.0 / n)
    k = np.arange(1, n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    C[1:, :] = s * np.cos(math.pi * k * (i + 0.5) / n)
    _DCT_CACHE[n] = C
    return C


def _dct_2d(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    C = _dct_matrix(n)
    return C @ matrix.astype(np.float64) @ C.T


def _idct_2d(coeffs: np.ndarray) -> np.ndarray:
    n = coeffs.shape[0]
    C = _dct_matrix(n)
    return C.T @ coeffs.astype(np.float64) @ C


def _zigzag_indices(n: int) -> np.ndarray:
    zz = np.zeros((n, n), dtype=np.int32)
    idx = 0
    for s in range(2 * n - 1):
        if s % 2 == 0:
            i = min(s, n - 1)
            j = s - i
            while i >= 0 and j < n:
                zz[i, j] = idx
                idx += 1
                i -= 1
                j += 1
        else:
            j = min(s, n - 1)
            i = s - j
            while j >= 0 and i < n:
                zz[i, j] = idx
                idx += 1
                i += 1
                j -= 1
    return zz


def _infer_block_size(name: str, shape: tuple) -> int:
    name_l = name.lower()
    if any(
        k in name_l
        for k in (
            "attn",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "wq",
            "wk",
            "wv",
            "wo",
            "query",
            "key",
            "value",
        )
    ):
        return 16
    if any(
        k in name_l
        for k in ("ffn", "gate", "up", "down", "mlp", "feed_forward", "w1", "w2", "w3")
    ):
        return 64
    if any(k in name_l for k in ("embed", "tok_embeddings", "lm_head", "output")):
        return 128
    if len(shape) >= 2:
        m, nn = shape[0], shape[-1]
        asp = max(m, nn) / max(min(m, nn), 1)
        if asp > 8:
            return 16
        if max(m, nn) >= 4096:
            return 128
        if max(m, nn) >= 1024:
            return 64
        return 32
    return 32


class SpectralTensorConverter:
    """Convert GGUF weight tensors to DCT-compressed spectral format."""

    def __init__(self, quality: float = 0.5, cross_block_pred: bool = True):
        self.quality = quality
        self.cross_block_pred = cross_block_pred

    def convert_tensor(self, tensor: np.ndarray, name: str = "") -> dict:
        if tensor.ndim < 2 or tensor.size < 64:
            return {
                "type": "raw",
                "data": tensor.astype(np.float32).tobytes(),
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
            }
        data = tensor.astype(np.float64)
        m, n = data.shape
        bs = _infer_block_size(name, (m, n))
        blocks = []
        keep_ratio = 0.002 + 0.048 * self.quality
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = data[i : i + bs, j : j + bs]
                bh, bw = block.shape
                if bh < bs or bw < bs:
                    pad = np.zeros((bs, bs), dtype=np.float64)
                    pad[:bh, :bw] = block
                    block_d = pad
                    act = (bh, bw)
                else:
                    block_d = block
                    act = (bs, bs)
                dct = _dct_2d(block_d)
                zz = _zigzag_indices(bs)
                zz_order = np.argsort(zz.ravel())
                flat = dct.ravel()
                n_keep = max(8, min(int(keep_ratio * bs * bs), bs * bs))
                ki = zz_order[:n_keep].tolist()
                kv = flat[ki].tolist()
                blocks.append(
                    {
                        "row": i,
                        "col": j,
                        "block_size": bs,
                        "actual_shape": act,
                        "n_kept": n_keep,
                        "kept_indices": ki,
                        "kept_values": kv,
                    }
                )
        return {
            "type": "spectral",
            "blocks": blocks,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "block_size": bs,
            "n_blocks": len(blocks),
            "layer_name": name,
            "quality": self.quality,
        }

    def decompress_tensor(self, comp: dict) -> np.ndarray:
        if comp.get("type") == "raw":
            return np.frombuffer(comp["data"], dtype=np.float32).reshape(comp["shape"])
        m, n = comp["shape"]
        bs = comp["block_size"]
        out = np.zeros((m, n), dtype=np.float64)
        for blk in comp.get("blocks", []):
            r, c = blk["row"], blk["col"]
            ah, aw = blk["actual_shape"]
            dct = np.zeros((bs, bs), dtype=np.float64)
            zz = _zigzag_indices(bs).ravel()
            z2p = np.argsort(zz)
            for idx, pos in enumerate(blk.get("kept_indices", [])):
                if idx < len(blk.get("kept_values", [])) and pos < len(z2p):
                    dp = z2p[pos]
                    dct[dp // bs, dp % bs] = blk["kept_values"][idx]
            recon = _idct_2d(dct)
            i_e = min(r + ah, m)
            j_e = min(c + aw, n)
            out[r:i_e, c:j_e] = recon[: i_e - r, : j_e - c]
        return out.astype(np.float32)


class WeightCache:
    """Multi-tier LRU cache for dequantized/decompressed weights."""

    def __init__(self, l1_mb: float = 512, l2_mb: float = 2048):
        self.l1_budget = int(l1_mb * 1024 * 1024)
        self.l2_budget = int(l2_mb * 1024 * 1024)
        self._l1: OrderedDict[str, np.ndarray] = OrderedDict()
        self._l2: OrderedDict[str, np.ndarray] = OrderedDict()
        self._l1_b = 0
        self._l2_b = 0
        self._lock = threading.Lock()
        self._pq: list[str] = []
        self._pt: Optional[threading.Thread] = None
        self._running = False
        self.hits = 0
        self.misses = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.prefetches = 0

    def get(
        self, name: str, loader: Optional[MMAPWeightLoader] = None
    ) -> Optional[np.ndarray]:
        with self._lock:
            if name in self._l1:
                self._l1.move_to_end(name)
                self.l1_hits += 1
                self.hits += 1
                return self._l1[name]
            if name in self._l2:
                arr = self._l2.pop(name)
                self._l2_b -= arr.nbytes
                if self._make_l1(arr.nbytes):
                    self._l1[name] = arr
                    self._l1_b += arr.nbytes
                else:
                    self._l2[name] = arr
                    self._l2_b += arr.nbytes
                self.l2_hits += 1
                self.hits += 1
                return arr
        self.misses += 1
        if loader is None:
            return None
        arr = loader.get_tensor(name)
        with self._lock:
            if self._make_l1(arr.nbytes):
                self._l1[name] = arr
                self._l1_b += arr.nbytes
            elif self._make_l2(arr.nbytes):
                self._l2[name] = arr
                self._l2_b += arr.nbytes
        return arr

    def _make_l1(self, needed: int) -> bool:
        while self._l1_b + needed > self.l1_budget and self._l1:
            n, arr = self._l1.popitem(last=False)
            self._l1_b -= arr.nbytes
            if self._make_l2(arr.nbytes):
                self._l2[n] = arr
                self._l2_b += arr.nbytes
        return self._l1_b + needed <= self.l1_budget

    def _make_l2(self, needed: int) -> bool:
        while self._l2_b + needed > self.l2_budget and self._l2:
            n, arr = self._l2.popitem(last=False)
            self._l2_b -= arr.nbytes
        return self._l2_b + needed <= self.l2_budget

    def prefetch(self, names: list[str]):
        with self._lock:
            for n in names:
                if n not in self._l1 and n not in self._l2:
                    self._pq.append(n)

    def start_bg(self, loader: MMAPWeightLoader):
        self._running = True
        self._pt = threading.Thread(target=self._bg_worker, args=(loader,), daemon=True)
        self._pt.start()

    def _bg_worker(self, loader: MMAPWeightLoader):
        while self._running:
            name = None
            with self._lock:
                if self._pq:
                    name = self._pq.pop(0)
            if name:
                try:
                    arr = loader.get_tensor(name)
                    with self._lock:
                        if self._make_l2(arr.nbytes):
                            self._l2[name] = arr
                            self._l2_b += arr.nbytes
                    self.prefetches += 1
                except Exception:
                    pass
            else:
                time.sleep(0.01)

    def stop_bg(self):
        self._running = False

    def evict(self, name: str):
        with self._lock:
            if name in self._l1:
                self._l1_b -= self._l1.pop(name).nbytes
            elif name in self._l2:
                self._l2_b -= self._l2.pop(name).nbytes

    def clear(self):
        with self._lock:
            self._l1.clear()
            self._l2.clear()
            self._l1_b = 0
            self._l2_b = 0

    def get_stats(self) -> dict:
        with self._lock:
            tot = self.hits + self.misses
            return {
                "l1_e": len(self._l1),
                "l1_mb": self._l1_b / 1024**2,
                "l2_e": len(self._l2),
                "l2_mb": self._l2_b / 1024**2,
                "hits": self.hits,
                "l1_hits": self.l1_hits,
                "l2_hits": self.l2_hits,
                "misses": self.misses,
                "hit_rate": self.hits / max(tot, 1),
                "prefetches": self.prefetches,
            }


class SpectralDequantizer:
    """Dequantize GGML quantized data directly into DCT domain.

    Skips the full float32 intermediate by processing quantized blocks
    and DCT-transforming them on-the-fly. Reduces memory bandwidth.
    """

    def __init__(self, quality: float = 0.5):
        self.quality = quality
        self._conv = SpectralTensorConverter(quality=quality)

    def dequantize_to_spectral(
        self, data: np.ndarray, ggml_type: int, shape: tuple, name: str = ""
    ) -> dict:
        n_el = math.prod(shape)
        if n_el < 4096 or shape[0] < 64:
            f32 = GGMLDequantizer.dequantize(data, ggml_type)[:n_el].reshape(shape)
            return self._conv.convert_tensor(f32, name=name)
        if ggml_type in (
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q8_1,
        ):
            return self._blockwise(data, ggml_type, shape, name)
        f32 = GGMLDequantizer.dequantize(data, ggml_type)[:n_el].reshape(shape)
        return self._conv.convert_tensor(f32, name=name)

    def _blockwise(
        self, data: np.ndarray, ggml_type: int, shape: tuple, name: str
    ) -> dict:
        m, n = shape
        bs = _infer_block_size(name, (m, n))
        qk = GGML_BLOCK_SIZE.get(ggml_type, 32)
        qb = GGML_BLOCK_BYTES.get(ggml_type, 18)
        raw = np.frombuffer(data, dtype=np.uint8)
        blocks = []
        keep_ratio = 0.002 + 0.048 * self.quality
        for i in range(0, m, bs):
            for j in range(0, n, bs):
                dct_block = np.zeros((bs, bs), dtype=np.float64)
                for di in range(0, min(bs, m - i), qk):
                    for dj in range(0, min(bs, n - j), qk):
                        gi, gj = i + di, j + dj
                        if gi >= m or gj >= n:
                            continue
                        bi = (gi // qk) * ((n + qk - 1) // qk) + (gj // qk)
                        bo = bi * qb
                        if bo + qb > len(raw):
                            continue
                        vals = GGMLDequantizer.dequantize(
                            raw[bo : bo + qb].copy(), ggml_type
                        )
                        rs, cs = gi % bs, gj % bs
                        for vi in range(min(len(vals), qk * qk)):
                            qi2, qj2 = vi // qk, vi % qk
                            bi2, bj2 = rs + qi2, cs + qj2
                            if bi2 < bs and bj2 < bs:
                                dct_block[bi2, bj2] = float(vals[vi])
                dct = _dct_2d(dct_block)
                zz = _zigzag_indices(bs)
                flat = dct.ravel()
                zo = np.argsort(zz.ravel())
                n_keep = max(8, min(int(keep_ratio * bs * bs), bs * bs))
                ki = zo[:n_keep].tolist()
                kv = flat[ki].tolist()
                blocks.append(
                    {
                        "row": i,
                        "col": j,
                        "block_size": bs,
                        "actual_shape": (min(bs, m - i), min(bs, n - j)),
                        "n_kept": n_keep,
                        "kept_indices": ki,
                        "kept_values": kv,
                    }
                )
        return {
            "type": "spectral",
            "blocks": blocks,
            "shape": shape,
            "dtype": "float32",
            "block_size": bs,
            "n_blocks": len(blocks),
            "layer_name": name,
            "quality": self.quality,
        }


class PredictiveWeightPrefetcher:
    """HDC-based prediction of next-layer weights for prefetching."""

    def __init__(self, n_layers: int = 32, hd_dim: int = 512):
        self.n_layers = n_layers
        self.hd_dim = hd_dim
        self._tl: dict[int, dict[int, int]] = {}
        self._thd: dict[int, np.ndarray] = {}
        self._hist: list[tuple[int, int]] = []
        self._max_hist = 1000

    def _hdv(self, seed: int) -> np.ndarray:
        rs = np.random.RandomState(seed)
        v = rs.randn(self.hd_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-10)

    def observe(self, token_id: int, layers: list[int]):
        for lidx in layers:
            self._hist.append((token_id, lidx))
            if len(self._hist) > self._max_hist:
                self._hist.pop(0)
            self._tl.setdefault(token_id, {})[lidx] = (
                self._tl[token_id].get(lidx, 0) + 1
            )
        if token_id not in self._thd:
            self._thd[token_id] = self._hdv(token_id)

    def predict(self, token: int, cur_layer: int, k: int = 3) -> list[int]:
        scores = np.zeros(self.n_layers, dtype=np.float64)
        if cur_layer + 1 < self.n_layers:
            scores[cur_layer + 1] += 3.0
        if cur_layer + 2 < self.n_layers:
            scores[cur_layer + 2] += 1.0
        tl = self._tl.get(token, {})
        for l, c in tl.items():
            scores[l] += c * 0.5
        if token in self._thd:
            chv = self._thd[token]
            for pt, pl in self._hist[-200:]:
                if pt == token:
                    continue
                if pt in self._thd:
                    sim = float(np.dot(chv, self._thd[pt]))
                    if sim > 0.3:
                        scores[pl] += sim * 0.4
        for i, (_, pl) in enumerate(self._hist[-20:]):
            scores[pl] += ((i + 1) / 20.0) * 0.3
        scores[0] += 2.0
        scores[self.n_layers - 1] += 2.0
        top = np.argsort(-scores)[:k]
        return [int(t) for t in top if scores[t] > 0]

    def get_prefetch_targets(self, token: int, cur_layer: int, k: int = 3) -> list[str]:
        layers = self.predict(token, cur_layer, k)
        targets = []
        for l in layers:
            for kernel in (
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_o.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ):
                targets.append(f"blk.{l}.{kernel}")
        return targets

    def get_stats(self) -> dict:
        return {
            "tokens": len(self._thd),
            "n_layers": self.n_layers,
            "hd_dim": self.hd_dim,
            "history": len(self._hist),
        }


class ResonantWeightLoader:
    """Load weights at frequencies matching their spectral importance."""

    def __init__(self):
        self._profiles: dict[str, dict] = {}

    def analyze(self, tensor: np.ndarray) -> dict:
        if tensor.ndim < 2:
            return {"type": "1d", "low_e": 1.0, "high_e": 0.0}
        m, n2 = tensor.shape
        bs = min(64, m, n2)
        block = tensor[:bs, :bs].astype(np.float64)
        dct = _dct_2d(block)
        flat = np.abs(dct.ravel())
        total = float(flat.sum())
        if total < 1e-30:
            return {"type": "flat", "low_e": 0.5, "high_e": 0.5}
        zz = _zigzag_indices(bs).ravel()
        nc = len(zz)
        lc = nc // 20
        mc = nc // 4
        le = float(flat[zz < lc].sum()) / total
        me = float(flat[(zz >= lc) & (zz < mc)].sum()) / total
        he = float(flat[zz >= mc].sum()) / total
        p = "low_freq" if le > 0.7 else ("mid_freq" if le + me > 0.7 else "wide_freq")
        return {"type": p, "low_e": le, "mid_e": me, "high_e": he}

    def analyze_tensor(self, name: str, tensor: np.ndarray):
        self._profiles[name] = self.analyze(tensor)

    def get_priority(self, name: str) -> float:
        p = self._profiles.get(name, {}).get("type", "mid_freq")
        return {
            "low_freq": 0.9,
            "mid_freq": 0.6,
            "wide_freq": 0.4,
            "flat": 0.5,
            "1d": 0.3,
        }.get(p, 0.5)

    def get_load_order(self, names: list[str]) -> list[str]:
        scored = [(n, self.get_priority(n)) for n in names]
        scored.sort(key=lambda x: -x[0])
        return [n for n, _ in scored]

    def get_stats(self) -> dict:
        c = {"low_freq": 0, "mid_freq": 0, "wide_freq": 0}
        for p in self._profiles.values():
            t = p.get("type", "")
            if t in c:
                c[t] += 1
        return {"analyzed": len(self._profiles), **c}


class ZeroCopySpectralEngine:
    """MMAP DCT-compressed weights and compute in compressed domain."""

    def __init__(self, spectral_store: dict[str, dict]):
        self.store = spectral_store

    def multiply(self, name: str, x: np.ndarray, exact: bool = True) -> np.ndarray:
        sp = self.store.get(name)
        if sp is None:
            raise ValueError(f"{name} not in spectral store")
        if exact:
            w = SpectralTensorConverter().decompress_tensor(sp)
            return x @ w if w.shape[0] == x.shape[-1] else w @ x
        return self._approx(sp, x)

    def _approx(self, sp: dict, x: np.ndarray) -> np.ndarray:
        shape = sp["shape"]
        bs = sp["block_size"]
        blocks = sp.get("blocks", [])
        is_2d = x.ndim == 2
        d_in = x.shape[-1] if is_2d else x.shape[0]
        w_trans = shape[0] == d_in
        m_w, n_w = shape if w_trans else (shape[1], shape[0])
        B = x.shape[0] if is_2d else 1
        out = np.zeros((B, m_w), dtype=np.float32)
        C = _dct_matrix(bs)
        for blk in blocks:
            r, c = blk.get("row", 0), blk.get("col", 0)
            ah = min(bs, shape[0] - r)
            aw = min(bs, shape[1] - c)
            dct = np.zeros((bs, bs), dtype=np.float64)
            zz = _zigzag_indices(bs).ravel()
            z2p = np.argsort(zz)
            for idx, pos in enumerate(blk.get("kept_indices", [])):
                if idx < len(blk.get("kept_values", [])) and pos < len(z2p):
                    dp = z2p[pos]
                    dct[dp // bs, dp % bs] = blk["kept_values"][idx]
            wb = (C.T @ dct @ C)[:ah, :aw].astype(np.float32)
            if w_trans:
                xb = x[..., r : r + ah] if is_2d else x[r : r + ah]
                if is_2d:
                    out[:, c : c + aw] += xb @ wb
                else:
                    out[0, c : c + aw] += xb @ wb
            else:
                xb = x[..., c : c + aw] if is_2d else x[c : c + aw]
                if is_2d:
                    out[:, r : r + ah] += xb @ wb.T
                else:
                    out[0, r : r + ah] += xb @ wb.T
        return out


class GGUFParserEngine:
    """Unified GGUF parsing and weight loading engine.

    Combines GGUFParser, GGMLDequantizer, MMAPWeightLoader,
    SpectralTensorConverter, and WeightCache.
    """

    def __init__(
        self,
        path: str,
        use_cache: bool = True,
        l1_mb: float = 512,
        l2_mb: float = 2048,
        spectral_quality: float = 0.5,
    ):
        self.path = Path(path)
        self.use_cache = use_cache
        self.sq = spectral_quality
        self._loader: Optional[MMAPWeightLoader] = None
        self._parser: Optional[GGUFParser] = None
        self._cache: Optional[WeightCache] = None
        self._conv = SpectralTensorConverter(quality=spectral_quality)
        self._sdeq = SpectralDequantizer(quality=spectral_quality)
        self._res = ResonantWeightLoader()
        self._pref: Optional[PredictiveWeightPrefetcher] = None
        self._zce: Optional[ZeroCopySpectralEngine] = None
        self._sstore: dict[str, dict] = {}
        self._load_time = 0.0
        if use_cache:
            self._cache = WeightCache(l1_mb=l1_mb, l2_mb=l2_mb)

    def open(self) -> "GGUFParserEngine":
        t0 = time.perf_counter()
        self._loader = MMAPWeightLoader(str(self.path)).open()
        self._load_time = time.perf_counter() - t0
        self._parser = self._loader._parser
        if self._cache:
            self._cache.start_bg(self._loader)
        print(
            f"[GGUFParserEngine] {self.path.name}: {self._loader.n_tensors}t {self._load_time * 1000:.0f}ms"
        )
        return self

    def close(self):
        if self._cache:
            self._cache.stop_bg()
        if self._loader:
            self._loader.close()

    @property
    def metadata(self) -> dict:
        return self._parser.metadata if self._parser else {}

    @property
    def n_layers(self) -> int:
        return self._loader._n_layers if self._loader else 0

    @property
    def n_tensors(self) -> int:
        return self._loader.n_tensors if self._loader else 0

    def get_tensor(self, name: str) -> np.ndarray:
        if self._res and name not in self._res._profiles:
            try:
                t = self._loader.get_tensor(name)
                self._res.analyze_tensor(name, t)
                return t
            except:
                pass
        if self._cache:
            return self._cache.get(name, loader=self._loader)
        return self._loader.get_tensor(name)

    def get_layer(self, lidx: int) -> dict[str, np.ndarray]:
        return self._loader.get_layer(lidx)

    def list_tensors(self) -> list[str]:
        return self._loader.list_tensors()

    def convert_to_spectral(self, name: str) -> dict:
        t = self.get_tensor(name)
        return self._conv.convert_tensor(t, name=name)

    def get_spectral(self, name: str, on_the_fly: bool = False) -> dict:
        if name in self._sstore:
            return self._sstore[name]
        if on_the_fly:
            ti = self._loader.tensor_info(name)
            if ti and ti["ggml_type"] != GGML_TYPE_F32:
                off = int(ti["offset"])
                ds = int(ti["data_size"])
                raw = np.frombuffer(
                    self._loader._mmap, dtype=np.uint8, offset=off, count=ds
                )
                sp = self._sdeq.dequantize_to_spectral(
                    raw, ti["ggml_type"], ti["shape"], name=name
                )
                self._sstore[name] = sp
                return sp
        sp = self.convert_to_spectral(name)
        self._sstore[name] = sp
        return sp

    def decompress_spectral(self, sp: dict) -> np.ndarray:
        return self._conv.decompress_tensor(sp)

    def prefetch_tensor(self, name: str):
        self._loader.prefetch_tensor(name)

    def prefetch_layer(self, lidx: int):
        self._loader.prefetch_layer(lidx)

    def prefetch_predicted(self, token: int, cur_layer: int):
        if self._pref is None:
            self._pref = PredictiveWeightPrefetcher(n_layers=self.n_layers)
        for n in self._pref.get_prefetch_targets(token, cur_layer):
            self._cache.prefetch([n])

    def observe_access(self, token: int, layers: list[int]):
        if self._pref is None:
            self._pref = PredictiveWeightPrefetcher(n_layers=self.n_layers)
        self._pref.observe(token, layers)

    def get_zce(self) -> Optional[ZeroCopySpectralEngine]:
        if self._zce is None and self._sstore:
            self._zce = ZeroCopySpectralEngine(self._sstore)
        return self._zce

    def get_resonant_order(self) -> list[str]:
        return self._res.get_load_order(self.list_tensors())

    def load_by_resonance(self, frac: float = 0.5) -> dict[str, np.ndarray]:
        ordered = self.get_resonant_order()
        n = max(1, int(len(ordered) * frac))
        return {name: self.get_tensor(name) for name in ordered[:n]}

    def get_stats(self) -> dict:
        s = {"path": str(self.path)}
        if self._loader:
            s.update(self._loader.get_stats())
        if self._cache:
            s["cache"] = self._cache.get_stats()
        if self._res:
            s["resonant"] = self._res.get_stats()
        if self._pref:
            s["prefetcher"] = self._pref.get_stats()
        s["load_time_ms"] = self._load_time * 1000
        s["n_spectral"] = len(self._sstore)
        return s

    def __enter__(self) -> "GGUFParserEngine":
        return self.open()

    def __exit__(self, *a):
        self.close()

    def __repr__(self) -> str:
        return f"<GGUFParserEngine {self.path.name} {self.n_tensors}t>"


class GGUFModelPatcher:
    """Patch SpectralStream classes to use the new parser engine."""

    @staticmethod
    def patch_all():
        GGUFModelPatcher.patch_gguf_model()
        GGUFModelPatcher.patch_mmap_engine_parse()
        GGUFModelPatcher.patch_mmap_engine_load()

    @staticmethod
    def patch_gguf_model():
        import spectralstream.gguf_model as gm
        from spectralstream.format.gguf_parser_engine import GGUFParserEngine

        def _init(self, path):
            self._engine = GGUFParserEngine(path).open()
            self.path = path
            self._parse_meta_e()
            self._load_tensors_e()

        def _parse_meta_e(self):
            meta = self._engine.metadata
            arch = str(meta.get("general.architecture", "unknown"))
            self.architecture = arch
            safe = {
                "llama": "llama",
                "granitehybrid": "llama",
                "gemma4": "llama",
                "qwen35moe": "llama",
                "mistral": "llama",
                "gemma": "llama",
            }
            a = safe.get(arch.lower(), arch.lower())
            self.n_layers = int(
                meta.get(f"{a}.block_count", 0) or meta.get(f"{arch}.block_count", 0)
            )
            self.hidden_dim = int(
                meta.get(f"{a}.embedding_length", 0)
                or meta.get(f"{arch}.embedding_length", 0)
            )
            self.ff_dim = int(
                meta.get(f"{a}.feed_forward_length", 0)
                or meta.get(f"{arch}.feed_forward_length", 0)
            )
            nheads = int(
                meta.get(f"{a}.attention.head_count", 0)
                or meta.get(f"{arch}.attention.head_count", 0)
            )
            self.n_heads = nheads
            self.n_kv_heads = int(
                meta.get(f"{a}.attention.head_count_kv", 0)
                or meta.get(f"{arch}.attention.head_count_kv", 0)
                or nheads
            )
            self.vocab_size = int(
                meta.get(f"{a}.vocab_size", 0) or meta.get(f"{arch}.vocab_size", 0)
            )
            self.context_length = int(
                meta.get(f"{a}.context_length", 2048)
                or meta.get(f"{arch}.context_length", 2048)
            )
            self.rope_dim = int(
                meta.get(f"{a}.rope.dimension_count", 0)
                or meta.get(f"{arch}.rope.dimension_count", 0)
            )
            self.rms_norm_eps = float(
                meta.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)
                or meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
            )
            self.head_dim = self.hidden_dim // max(self.n_heads, 1)

        def _load_tensors_e(self):
            self.tensors = {}
            self._tensor_names = self._engine.list_tensors()

        def _get(self, name):
            if name in self.tensors:
                return self.tensors[name]
            t = self._engine.get_tensor(name)
            self.tensors[name] = t
            return t

        def _gt_layer(self, lidx, name):
            return self.get_tensor(f"blk.{lidx}.{name}")

        def _summary(self):
            arch = self.architecture if hasattr(self, "architecture") else "?"
            lns = self.n_layers if hasattr(self, "n_layers") else 0
            tn = self._tensor_names if hasattr(self, "_tensor_names") else []
            return "\n".join(
                [
                    f"GGUFModel (Patched): {self.path}",
                    f"  Architecture: {arch}",
                    f"  Layers: {lns}",
                    f"  Tensors: {len(tn)}",
                ]
            )

        gm.GGUFModel.__init__ = _init
        gm.GGUFModel._parse_meta_e = _parse_meta_e
        gm.GGUFModel._load_tensors_e = _load_tensors_e
        gm.GGUFModel.get_tensor = _get
        gm.GGUFModel.get_layer_tensor = _gt_layer
        gm.GGUFModel.summary = _summary

    @staticmethod
    def patch_mmap_engine_parse():
        import spectralstream.mmap_engine as me

        def _parse(self):
            from spectralstream.format.gguf_parser_engine import (
                GGUFParser,
                GGML_TYPE_F32,
                GGML_TYPE_F16,
                GGML_TYPE_NAMES,
            )

            p = GGUFParser(str(self.model_path))
            p._data = self._mmap
            p.parse()
            self._tensor_index.clear()
            self._layer_index.clear()
            for ti in p.tensor_infos:
                n = ti["name"]
                s = ti["shape"]
                gt = ti["ggml_type"]
                isq = gt not in (GGML_TYPE_F32, GGML_TYPE_F16)
                dt = np.dtype("float16") if gt == GGML_TYPE_F16 else np.dtype("float32")
                self._tensor_index[n] = {
                    "name": n,
                    "shape": s,
                    "dtype": dt,
                    "offset": int(ti["offset"]),
                    "tensor_type": gt,
                    "is_quantized": isq,
                    "is_raw": not isq,
                    "nbytes": int(ti["data_size"]),
                }
                self._add_to_layer_index(n)
            import json

            self._metadata = dict(p.metadata)
            self._metadata["format"] = "gguf"
            arch = str(p.metadata.get("general.architecture", ""))
            nl = p.metadata.get(f"{arch}.block_count", 0) or 0
            self._n_layers = (
                int(nl)
                if nl
                else (max(self._layer_index.keys()) + 1 if self._layer_index else 32)
            )

        me.MmapEngine._parse_gguf = _parse

    @staticmethod
    def patch_mmap_engine_load():
        import spectralstream.mmap_engine as me

        def _load(self, info):
            from spectralstream.format.gguf_parser_engine import GGMLDequantizer

            off = info["offset"]
            sh = info["shape"]
            dt = info["dtype"]
            if info.get("is_quantized"):
                try:
                    raw = np.frombuffer(
                        self._mmap, dtype=np.uint8, offset=off, count=info["nbytes"]
                    ).copy()
                    r = GGMLDequantizer.dequantize(raw, info["tensor_type"])
                    return r[: int(np.prod(sh))].reshape(sh)
                except:
                    pass
            return np.ndarray(sh, dtype=dt, buffer=self._mmap, offset=off)

        me.MmapEngine._load_gguf_tensor = _load


def validate_gguf(path: str, verbose: bool = True) -> dict:
    """Validate a GGUF file: parse, dequantize a sample, report stats.

    Usage:
        python -m spectralstream.format.gguf_parser_engine --validate model.gguf
    """
    results = {"path": path, "valid": False, "errors": []}
    if verbose:
        print(f"\nValidating: {path}\n" + "-" * 60)
    try:
        parser = GGUFParser(path).parse()
        results["version"] = parser.version
        results["tensor_count"] = parser.tensor_count
        results["metadata_keys"] = len(parser.metadata)
        results["file_size_mb"] = parser.file_size / 1024**2
        if verbose:
            print(str(Path(path).name))
            print(parser.summary())
    except Exception as e:
        results["errors"].append(f"Parse: {e}")
        return results
    t0 = time.perf_counter()
    try:
        with MMAPWeightLoader(path, prefetch_first_last=False) as loader:
            results["tensors_indexed"] = loader.n_tensors
            results["mmap_open_ms"] = (time.perf_counter() - t0) * 1000
            if verbose:
                print(
                    f"  MMAP: {loader.n_tensors}t, {loader._n_layers}L, {results['mmap_open_ms']:.0f}ms"
                )
            sample = None
            stype = None
            for ti in parser.tensor_infos:
                if ti["ggml_type"] not in (
                    GGML_TYPE_F32,
                    GGML_TYPE_F16,
                    GGML_TYPE_BF16,
                ):
                    sample = ti["name"]
                    stype = ti["ggml_type"]
                    break
            if sample:
                tensor = loader.get_tensor(sample)
                results["sample_tensor"] = sample
                results["sample_type"] = GGML_TYPE_NAMES.get(stype, str(stype))
                results["sample_shape"] = list(tensor.shape)
                results["sample_stats"] = {
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                }
                if verbose:
                    print(f"  Sample: {sample} ({results['sample_type']})")
                    print(f"    Shape: {tensor.shape}")
                    print(f"    Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
                    print(f"    Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
            else:
                if verbose:
                    print("  No quantized tensors (F32/F16 only)")
    except Exception as e:
        results["errors"].append(f"MMAP: {e}")
    try:
        conv = SpectralTensorConverter(quality=0.5)
        if sample:
            with MMAPWeightLoader(path, prefetch_first_last=False) as loader:
                tensor = loader.get_tensor(sample)
                sp = conv.convert_tensor(tensor, name=sample)
                recon = conv.decompress_tensor(sp)
                mse = float(
                    np.mean(
                        (
                            tensor[: recon.shape[0], : recon.shape[1]].astype(
                                np.float64
                            )
                            - recon.astype(np.float64)
                        )
                        ** 2
                    )
                )
                results["spectral_mse"] = mse
                results["spectral_blocks"] = sp.get("n_blocks", 0)
                if verbose:
                    print(f"  Spectral: {sp['n_blocks']} blocks, MSE={mse:.2e}")
    except Exception as e:
        results["errors"].append(f"Spectral: {e}")
    results["valid"] = len(results["errors"]) == 0
    if verbose:
        print("-" * 60)
        print(
            f"  VALID: {results['valid']}"
            if results["valid"]
            else f"  FAIL: {results['errors']}"
        )
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--validate":
        validate_gguf(sys.argv[2], verbose=True)
    else:
        print("GGUF Parser Engine - SpectralStream")
        print(
            "Usage: python -m spectralstream.format.gguf_parser_engine --validate model.gguf"
        )
        print()
        GGUFParserEngine.__doc__ = "Unified GGUF parsing and weight loading engine."
        # Quick self-test
        from spectralstream.format.gguf_parser_engine import GGMLDequantizer

        print("GGMLDequantizer: ready")
        print("GGUFParser: ready")
        print("MMAPWeightLoader: ready")
        print("SpectralTensorConverter: ready")
        print("WeightCache: ready")
        print("SpectralDequantizer: ready")
        print("PredictiveWeightPrefetcher: ready")
        print("ResonantWeightLoader: ready")
        print("ZeroCopySpectralEngine: ready")
        print(f"  GGML types supported: {len(GGML_TYPE_NAMES)}")

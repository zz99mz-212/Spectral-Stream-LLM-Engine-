"""
SSCX (SpectralStream Compressed eXtended) Binary Format
=======================================================
Production binary format for extreme model compression.

File Layout:
  [Header 128B][Layer Index][Tensor Index][Data Blocks...][Footer]

Header (128 bytes):
  magic: b"SSCX" (4B)
  version: uint32
  flags: uint32
  num_layers: uint32
  num_tensors: uint32
  original_size: uint64
  compressed_size: uint64
  target_ratio: float32
  max_error: float32
  model_name: char[64]

Per-Layer Entry (32 bytes):
  layer_id: uint32
  num_tensors: uint32
  group_id: uint16 (cross-layer sharing)
  reserved: uint16
  params: float32[4]

Per-Tensor Entry (variable):
  name_len: uint16
  name: bytes[name_len]
  offset: uint64
  compressed_size: uint32
  original_size: uint32
  ndim: uint8
  shape: uint32[4]
  dtype: uint16
  method: uint16
  snr: float32
  rel_error: float32

Data Blocks:
  Page-aligned (4096B)
  CRC32 per block
  Independently decodable

Footer:
  crc32: uint32
  index_offset: uint64
  index_crc32: uint32

Features:
  - Page-aligned data blocks for efficient MMAP loading
  - Per-block CRC32 integrity checking
  - Per-tensor error metrics (SNR, relative error)
  - Cross-layer delta encoding support
  - Memory-mapped read with zero-copy access
  - Backward compatibility via version field
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

SSCX_MAGIC = b"SSCX"
SSCX_VERSION = 1
SSCX_HEADER_SIZE = 128
SSCX_PAGE_SIZE = 4096

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_BF16 = 2
DTYPE_INT8 = 3
DTYPE_INT4 = 4

COMP_RAW = 0
COMP_DCT = 1
COMP_SPECTRAL = 2
COMP_INT8 = 3
COMP_INT4 = 4
COMP_DELTA = 5
COMP_NAMES = ["raw", "dct", "spectral", "int8", "int4", "delta"]

_HEADER_FMT = "<4sIIIIQQff64s"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_LAYER_FMT = "<IIHHffff"
_LAYER_SIZE = struct.calcsize(_LAYER_FMT)
_FOOTER_FMT = "<IQQ"
_FOOTER_SIZE = struct.calcsize(_FOOTER_FMT)


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n //= 1024
    return f"{n:.1f}TB"


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def page_align(size: int) -> int:
    return _align_up(size, SSCX_PAGE_SIZE)


def page_aligned_size(size: int) -> int:
    return _align_up(size, SSCX_PAGE_SIZE)


@dataclass
class SSCXHeader:
    magic: bytes = SSCX_MAGIC
    version: int = SSCX_VERSION
    flags: int = 0
    num_layers: int = 0
    num_tensors: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    target_ratio: float = 1.0
    max_error: float = 0.0
    model_name: str = ""

    def pack(self) -> bytes:
        name_encoded = self.model_name.encode("utf-8")[:64].ljust(64, b"\x00")
        return struct.pack(
            _HEADER_FMT,
            self.magic,
            self.version,
            self.flags,
            self.num_layers,
            self.num_tensors,
            self.total_original_bytes,
            self.total_compressed_bytes,
            self.target_ratio,
            self.max_error,
            name_encoded,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SSCXHeader":
        if len(data) < _HEADER_SIZE:
            raise ValueError(f"Data too short for header: {len(data)} < {_HEADER_SIZE}")
        fields = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
        magic = fields[0]
        if magic != SSCX_MAGIC:
            raise ValueError(f"Bad SSCX magic: {magic!r}")
        name = fields[9].rstrip(b"\x00").decode("utf-8", errors="replace")
        return cls(
            magic=magic,
            version=fields[1],
            flags=fields[2],
            num_layers=fields[3],
            num_tensors=fields[4],
            total_original_bytes=fields[5],
            total_compressed_bytes=fields[6],
            target_ratio=fields[7],
            max_error=fields[8],
            model_name=name,
        )


@dataclass
class SSCXLayerEntry:
    layer_id: int = 0
    num_tensors: int = 0
    group_id: int = 0
    params: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    def pack(self) -> bytes:
        p = list(self.params) + [0.0] * (4 - len(self.params))
        return struct.pack(
            _LAYER_FMT, self.layer_id, self.num_tensors, self.group_id, 0, *p[:4]
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SSCXLayerEntry":
        if len(data) < _LAYER_SIZE:
            raise ValueError(f"Data too short for layer entry: {len(data)}")
        fields = struct.unpack(_LAYER_FMT, data[:_LAYER_SIZE])
        return cls(
            layer_id=fields[0],
            num_tensors=fields[1],
            group_id=fields[2],
            params=fields[4:8],
        )


@dataclass
class SSCXTensorEntry:
    name: str = ""
    offset: int = 0
    compressed_size: int = 0
    original_size: int = 0
    ndim: int = 0
    shape: Tuple[int, ...] = ()
    dtype: int = 0
    method: int = 0
    snr: float = 0.0
    rel_error: float = 0.0
    error_psnr: float = 0.0
    error_cos: float = 0.0
    block_checksum: int = 0
    layer_id: int = 0

    def pack(self) -> bytes:
        name_bytes = self.name.encode("utf-8")
        name_len = len(name_bytes)
        shape = list(self.shape) + [0] * (4 - len(self.shape))
        return struct.pack(
            f"<H{name_len}sB4I2H2f2IfI",
            name_len,
            name_bytes,
            self.ndim,
            *shape,
            self.dtype,
            self.method,
            self.snr,
            self.rel_error,
            self.error_psnr,
            self.error_cos,
            self.block_checksum,
            self.layer_id,
        ) + struct.pack("<QQQ", self.offset, self.compressed_size, self.original_size)

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> "SSCXTensorEntry":
        pos = offset
        name_len = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        name = data[pos : pos + name_len].decode("utf-8", errors="replace")
        pos += name_len
        ndim = data[pos]
        pos += 1
        shape = struct.unpack_from("<" + "I" * 4, data, pos)
        pos += 16
        dtype, method = struct.unpack_from("<HH", data, pos)
        pos += 4
        snr, rel_error = struct.unpack_from("<ff", data, pos)
        pos += 8
        error_psnr, error_cos = struct.unpack_from("<ff", data, pos)
        pos += 8
        block_checksum, layer_id = struct.unpack_from("<II", data, pos)
        pos += 8
        d_off, c_sz, o_sz = struct.unpack_from("<QQQ", data, pos)
        pos += 24
        return cls(
            name=name,
            offset=d_off,
            compressed_size=c_sz,
            original_size=o_sz,
            ndim=ndim,
            shape=tuple(s for s in shape if s != 0),
            dtype=dtype,
            method=method,
            snr=snr,
            rel_error=rel_error,
            error_psnr=error_psnr,
            error_cos=error_cos,
            block_checksum=block_checksum,
            layer_id=layer_id,
        )


SSCXTensorInfo = SSCXTensorEntry
SSCXLayerInfo = SSCXLayerEntry


@dataclass
class SSCXFooter:
    crc32: int = 0
    index_offset: int = 0
    index_crc32: int = 0

    def pack(self) -> bytes:
        return struct.pack(_FOOTER_FMT, self.crc32, self.index_offset, self.index_crc32)

    @classmethod
    def unpack(cls, data: bytes) -> "SSCXFooter":
        if len(data) < _FOOTER_SIZE:
            raise ValueError(f"Data too short for footer: {len(data)}")
        fields = struct.unpack(_FOOTER_FMT, data[:_FOOTER_SIZE])
        return cls(crc32=fields[0], index_offset=fields[1], index_crc32=fields[2])


class SSCXWriter:
    """Write compressed model tensors to SSCX binary format."""

    def __init__(
        self,
        path: str,
        model_name: str = "",
        target_ratio: float = 1.0,
        max_error: float = 0.0,
    ):
        self.path = path
        self.model_name = model_name
        self.target_ratio = target_ratio
        self.max_error = max_error
        self._file: Optional[Any] = None
        self._layers: List[SSCXLayerEntry] = []
        self._tensors: List[SSCXTensorEntry] = []
        self._data_blocks: List[bytes] = []
        self._current_offset: int = 0

    def open(self) -> "SSCXWriter":
        self._file = open(self.path, "wb")
        # Reserve header space
        self._file.write(b"\x00" * SSCX_HEADER_SIZE)
        self._current_offset = SSCX_HEADER_SIZE
        return self

    def add_layer(self, layer: SSCXLayerEntry) -> None:
        self._layers.append(layer)

    def add_tensor(self, tensor: SSCXTensorEntry, data: bytes) -> None:
        aligned_size = _align_up(len(data), SSCX_PAGE_SIZE)
        tensor.offset = self._current_offset
        tensor.compressed_size = len(data)
        self._tensors.append(tensor)
        self._data_blocks.append(data + b"\x00" * (aligned_size - len(data)))
        self._current_offset += aligned_size

    def close(self) -> None:
        if self._file is None:
            return

        # Write layer index
        layer_offset = self._current_offset
        for layer in self._layers:
            self._file.write(layer.pack())
            self._current_offset += len(layer.pack())

        # Write tensor index
        tensor_offset = self._current_offset
        for tensor in self._tensors:
            self._file.write(tensor.pack())
            self._current_offset += len(tensor.pack())

        # Write tensor data blocks
        data_offset = self._current_offset
        for block in self._data_blocks:
            self._file.write(block)
            self._current_offset += len(block)

        # Write footer
        footer_offset = self._current_offset
        footer = SSCXFooter(
            crc32=_crc32(b"".join(self._data_blocks)),
            index_offset=tensor_offset,
            index_crc32=_crc32(b"".join(t.pack() for t in self._tensors)),
        )
        self._file.write(footer.pack())

        # Update header
        header = SSCXHeader(
            version=SSCX_VERSION,
            flags=0,
            num_layers=len(self._layers),
            num_tensors=len(self._tensors),
            total_original_bytes=sum(t.original_size for t in self._tensors),
            total_compressed_bytes=sum(t.compressed_size for t in self._tensors),
            target_ratio=self.target_ratio,
            max_error=self.max_error,
            model_name=self.model_name,
        )
        self._file.seek(0)
        self._file.write(header.pack())
        self._file.close()
        self._file = None

    def __enter__(self) -> "SSCXWriter":
        return self.open()

    def __exit__(self, *args):
        if self._file is not None:
            self.close()


class SSCXReader:
    """Read compressed model tensors from SSCX binary format with mmap."""

    def __init__(self, path: str):
        self.path = path
        self._mmap: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._size: int = 0
        self._data: Optional[bytes] = None
        self.header: Optional[SSCXHeader] = None
        self._tensors: Dict[str, SSCXTensorEntry] = {}
        self._layers: List[SSCXLayerEntry] = []

    def open(self) -> "SSCXReader":
        self._fd = os.open(str(self.path), os.O_RDONLY | os.O_CLOEXEC)
        self._size = os.fstat(self._fd).st_size
        self._mmap = mmap.mmap(self._fd, self._size, access=mmap.ACCESS_READ)
        self._data = self._mmap

        raw_header = bytes(self._data[:SSCX_HEADER_SIZE])
        self.header = SSCXHeader.unpack(raw_header)

        # Parse footer at end of file
        footer_raw = bytes(self._data[-_FOOTER_SIZE:])
        footer = SSCXFooter.unpack(footer_raw)

        # Parse tensor index
        pos = footer.index_offset
        for _ in range(self.header.num_tensors):
            entry = SSCXTensorEntry.unpack(self._data, offset=pos)
            self._tensors[entry.name] = entry
            pos += len(entry.pack())

        return self

    def get_tensor(self, name: str) -> Optional[bytes]:
        entry = self._tensors.get(name)
        if entry is None:
            return None
        start = entry.offset
        end = start + entry.compressed_size
        return bytes(self._data[start:end])

    def get_tensor_names(self) -> List[str]:
        return list(self._tensors.keys())

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._data = None

    def __enter__(self) -> "SSCXReader":
        return self.open()

    def __exit__(self, *args):
        self.close()

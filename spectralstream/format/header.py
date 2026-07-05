from __future__ import annotations

import struct
from typing import Any

from spectralstream.format.core import (
    SSF_MAGIC,
    SSF_HEADER_SIZE,
    SSF_REDUNDANT_HEADER_OFFSET,
)


class SSFHeader:
    FORMAT = "<4sIIIQQQQQQQ184s"

    def __init__(
        self,
        magic: bytes = SSF_MAGIC,
        version: int = 3,
        flags: int = 0,
        n_tensors: int = 0,
        index_offset: int = 0,
        index_size: int = 0,
        metadata_offset: int = 0,
        metadata_size: int = 0,
        tensor_data_offset: int = 0,
        redundant_header_offset: int = SSF_REDUNDANT_HEADER_OFFSET,
        footer_offset: int = 0,
    ):
        self.magic = magic
        self.version = version
        self.flags = flags
        self.n_tensors = n_tensors
        self.index_offset = index_offset
        self.index_size = index_size
        self.metadata_offset = metadata_offset
        self.metadata_size = metadata_size
        self.tensor_data_offset = tensor_data_offset
        self.redundant_header_offset = redundant_header_offset
        self.footer_offset = footer_offset

    def pack(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.magic,
            self.version,
            self.flags,
            self.n_tensors,
            self.index_offset,
            self.index_size,
            self.metadata_offset,
            self.metadata_size,
            self.tensor_data_offset,
            self.redundant_header_offset,
            self.footer_offset,
            b"\x00" * 184,
        )

    @classmethod
    def unpack(cls, data: bytes) -> SSFHeader:
        if len(data) < SSF_HEADER_SIZE:
            raise ValueError(f"Header too small: {len(data)} < {SSF_HEADER_SIZE}")
        fields = struct.unpack(cls.FORMAT, data[:SSF_HEADER_SIZE])
        magic = fields[0]
        ver = fields[1]
        if magic != SSF_MAGIC:
            raise ValueError(f"Bad magic: {magic!r} (expected {SSF_MAGIC!r})")
        if ver not in (2, 3):
            raise ValueError(f"Unsupported SSF version: {ver}")
        return cls(
            magic=magic,
            version=ver,
            flags=fields[2],
            n_tensors=fields[3],
            index_offset=fields[4],
            index_size=fields[5],
            metadata_offset=fields[6],
            metadata_size=fields[7],
            tensor_data_offset=fields[8],
            redundant_header_offset=fields[9],
            footer_offset=fields[10],
        )

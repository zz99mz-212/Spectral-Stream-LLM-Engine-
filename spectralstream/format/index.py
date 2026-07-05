from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from spectralstream.format.core import TensorDType, _LEGACY_COMPRESSION_MAP


@dataclass
class TensorIndexEntry:
    name: str
    shape: Tuple[int, ...]
    dtype: TensorDType
    compression_method: int
    compression_params: dict
    data_offset: int
    compressed_size: int
    original_size: int
    quality_metrics: Dict[str, float]
    checksum: bytes
    flags: int = 0

    def pack(self) -> bytes:
        name_bytes = self.name.encode("utf-8")
        buf = struct.pack("<I", len(name_bytes))
        buf += name_bytes
        buf += struct.pack("<I", len(self.shape))
        for d in self.shape:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<H", self.dtype.value)
        buf += struct.pack("<i", self.compression_method)
        params_json = (
            json.dumps(self.compression_params, sort_keys=True).encode("utf-8")
            if self.compression_params
            else b"{}"
        )
        buf += struct.pack("<I", len(params_json))
        buf += params_json
        buf += struct.pack(
            "<QQQ", self.data_offset, self.compressed_size, self.original_size
        )
        q_flat = [
            self.quality_metrics.get(k, 0.0)
            for k in (
                "relative_error",
                "snr_db",
                "psnr_db",
                "cosine_similarity",
                "compression_ratio",
            )
        ]
        for v in q_flat:
            buf += struct.pack("<d", v)
        buf += self.checksum[:32].ljust(32, b"\x00")
        buf += struct.pack("<I", self.flags)
        return buf

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> Tuple[TensorIndexEntry, int]:
        pos = offset
        name_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + name_len > len(data):
            raise ValueError("Truncated index entry name")
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        ndim = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + ndim * 8 > len(data):
            raise ValueError("Truncated index entry shape")
        shape = tuple(struct.unpack_from("<" + "Q" * ndim, data, pos))
        pos += ndim * 8
        dtype_val = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        method_id = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        params_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if params_len > 0 and pos + params_len <= len(data):
            try:
                params = json.loads(data[pos : pos + params_len])
            except (json.JSONDecodeError, UnicodeDecodeError):
                params = {}
        else:
            params = {}
        pos += params_len
        data_off, comp_sz, orig_sz = struct.unpack_from("<QQQ", data, pos)
        pos += 24
        q_keys = (
            "relative_error",
            "snr_db",
            "psnr_db",
            "cosine_similarity",
            "compression_ratio",
        )
        q_values = struct.unpack_from("<5d", data, pos)
        pos += 40
        quality_metrics = dict(zip(q_keys, q_values))
        checksum = data[pos : pos + 32]
        pos += 32
        flags = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        entry = cls(
            name=name,
            shape=shape,
            dtype=TensorDType(dtype_val),
            compression_method=method_id,
            compression_params=params,
            data_offset=data_off,
            compressed_size=comp_sz,
            original_size=orig_sz,
            quality_metrics=quality_metrics,
            checksum=checksum,
            flags=flags,
        )
        return entry, pos - offset


class LegacyTensorIndexEntry:
    __slots__ = (
        "name",
        "shape",
        "dtype",
        "compression",
        "data_offset",
        "compressed_size",
        "original_size",
        "checksum",
        "flags",
    )

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: TensorDType,
        compression: int,
        data_offset: int,
        compressed_size: int,
        original_size: int,
        checksum: bytes,
        flags: int = 0,
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.compression = compression
        self.data_offset = data_offset
        self.compressed_size = compressed_size
        self.original_size = original_size
        self.checksum = checksum
        self.flags = flags

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> Tuple[LegacyTensorIndexEntry, int]:
        pos = offset
        name_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        ndim = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        shape = tuple(struct.unpack_from("<" + "Q" * ndim, data, pos))
        pos += ndim * 8
        dtype_val, comp_val = struct.unpack_from("<HH", data, pos)
        pos += 4
        data_off, comp_sz, orig_sz = struct.unpack_from("<QQQ", data, pos)
        pos += 24
        checksum = data[pos : pos + 32]
        pos += 32
        flags = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return cls(
            name=name,
            shape=shape,
            dtype=TensorDType(dtype_val),
            compression=comp_val,
            data_offset=data_off,
            compressed_size=comp_sz,
            original_size=orig_sz,
            checksum=checksum,
            flags=flags,
        ), pos - offset


class TensorIndex:
    def __init__(self) -> None:
        self._entries: list[TensorIndexEntry] = []
        self._by_name: dict[str, TensorIndexEntry] = {}

    @property
    def entries(self) -> List[TensorIndexEntry]:
        return self._entries

    def add(self, entry: TensorIndexEntry) -> None:
        self._entries.append(entry)
        self._by_name[entry.name] = entry

    def get(self, name: str) -> Optional[TensorIndexEntry]:
        return self._by_name.get(name)

    def __getitem__(self, idx: int) -> TensorIndexEntry:
        return self._entries[idx]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[TensorIndexEntry]:
        return iter(self._entries)

    def names(self) -> list[str]:
        return [e.name for e in self._entries]

    def pack(self) -> bytes:
        buf = bytearray()
        for e in self._entries:
            buf += e.pack()
        return bytes(buf)

    @classmethod
    def unpack(cls, data: bytes, is_legacy: bool = False) -> TensorIndex:
        idx = cls()
        pos = 0
        if is_legacy:
            while pos < len(data):
                entry, size = LegacyTensorIndexEntry.unpack(data, pos)
                qm = {
                    "relative_error": 0.0,
                    "snr_db": 0.0,
                    "psnr_db": 0.0,
                    "cosine_similarity": 0.0,
                    "compression_ratio": 0.0,
                }
                method_id = _LEGACY_COMPRESSION_MAP.get(entry.compression, 0)
                new_entry = TensorIndexEntry(
                    name=entry.name,
                    shape=entry.shape,
                    dtype=entry.dtype,
                    compression_method=method_id,
                    compression_params={},
                    data_offset=entry.data_offset,
                    compressed_size=entry.compressed_size,
                    original_size=entry.original_size,
                    quality_metrics=qm,
                    checksum=entry.checksum,
                    flags=entry.flags,
                )
                idx.add(new_entry)
                pos += size
        else:
            while pos < len(data):
                entry, size = TensorIndexEntry.unpack(data, pos)
                idx.add(entry)
                pos += size
        return idx

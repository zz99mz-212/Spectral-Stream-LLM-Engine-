from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ModelScanResult:
    path: str
    format: str
    total_bytes: int = 0
    tensor_count: int = 0
    tensors: Dict[str, Tuple[Tuple[int, ...], str, int, int]] = field(
        default_factory=dict
    )
    dtype_distribution: Dict[str, int] = field(default_factory=dict)
    shape_categories: Dict[str, int] = field(default_factory=dict)


class ModelScanner:
    SAFETENSORS_DTYPE_MAP: Dict[str, str] = {
        "F32": "float32",
        "F16": "float16",
        "BF16": "bfloat16",
        "I64": "int64",
        "I32": "int32",
        "I16": "int16",
        "I8": "int8",
        "U8": "uint8",
    }

    def scan(self, path: str) -> ModelScanResult:
        path_lower = path.lower()
        if path_lower.endswith(".safetensors"):
            return self._scan_safetensors(path)
        if path_lower.endswith(".gguf"):
            return self._scan_gguf(path)
        if path_lower.endswith(".ssf"):
            return self._scan_ssf(path)
        raise ValueError(f"Unknown model format: {path}")

    def _scan_safetensors(self, path: str) -> ModelScanResult:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        result = ModelScanResult(path=path, format="safetensors")
        dtype_dist: Dict[str, int] = {}
        shape_cats: Dict[str, int] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info.get("dtype", "F32")
            shape = tuple(info.get("shape", []))
            offsets = info.get("data_offsets", [0, 0])
            nbytes = offsets[1] - offsets[0]
            result.tensors[name] = (shape, dtype_str, data_start + offsets[0], nbytes)
            result.total_bytes += nbytes
            dtype_dist[dtype_str] = dtype_dist.get(dtype_str, 0) + 1
            cat = (
                f"{len(shape)}d_{shape[0]}x{shape[1]}"
                if len(shape) >= 2
                else (f"{len(shape)}d_{shape[0]}" if shape else "scalar")
            )
            shape_cats[cat] = shape_cats.get(cat, 0) + 1
        result.tensor_count = len(result.tensors)
        result.dtype_distribution = dtype_dist
        result.shape_categories = shape_cats
        return result

    def _scan_gguf(self, path: str) -> ModelScanResult:
        try:
            from spectralstream.format.gguf_parser_engine import GGUFParser
        except ImportError:
            raise ImportError("GGUF parser engine not available")
        parser = GGUFParser(path)
        parser.parse()
        result = ModelScanResult(path=path, format="gguf")
        dtype_dist: Dict[str, int] = {}
        shape_cats: Dict[str, int] = {}
        for ti in parser.tensor_infos:
            name = ti["name"]
            shape = tuple(ti.get("shape", []))
            raw_dtype = ti.get("ggml_type", "F32")
            dtype_str = str(raw_dtype)
            nbytes = ti.get("data_size", 0)
            result.tensors[name] = (shape, dtype_str, ti.get("offset", 0), nbytes)
            result.total_bytes += nbytes
            dtype_dist[dtype_str] = dtype_dist.get(dtype_str, 0) + 1
            cat = (
                f"{len(shape)}d_{shape[0]}x{shape[1]}"
                if len(shape) >= 2
                else (f"{len(shape)}d_{shape[0]}" if shape else "scalar")
            )
            shape_cats[cat] = shape_cats.get(cat, 0) + 1
        result.tensor_count = len(result.tensors)
        result.dtype_distribution = dtype_dist
        result.shape_categories = shape_cats
        return result

    def _scan_ssf(self, path: str) -> ModelScanResult:
        try:
            from spectralstream.format.ssf_format import SSFReader
        except ImportError:
            raise ImportError("SSF format not available")
        reader = SSFReader(path, mmap_mode=True)
        result = ModelScanResult(path=path, format="ssf")
        dtype_dist: Dict[str, int] = {}
        shape_cats: Dict[str, int] = {}
        index = reader._index
        if index is None:
            reader.close()
            return result
        for e in index:
            name = e.name
            shape = e.shape
            dtype_str = e.dtype.name
            nbytes = e.original_size
            result.total_bytes += nbytes
            result.tensors[name] = (shape, dtype_str, e.data_offset, e.compressed_size)
            dtype_dist[dtype_str] = dtype_dist.get(dtype_str, 0) + 1
            cat = (
                f"{len(shape)}d_{shape[0]}x{shape[1]}"
                if len(shape) >= 2
                else (f"{len(shape)}d_{shape[0]}" if shape else "scalar")
            )
            shape_cats[cat] = shape_cats.get(cat, 0) + 1
        result.tensor_count = len(result.tensors)
        result.dtype_distribution = dtype_dist
        result.shape_categories = shape_cats
        reader.close()
        return result

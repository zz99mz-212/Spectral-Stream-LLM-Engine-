from __future__ import annotations

import importlib
import threading
import zlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.format.core import _LOSSY_TO_METHOD, _LEGACY_COMPRESSION_MAP

_ENGINE_METHODS: dict = {}
_ENGINE_LOCK = threading.Lock()


def _import_method_registry() -> Any:
    try:
        from spectralstream.compression.method_registry import (
            CompressionMethod as RegistryMethod,
        )

        return RegistryMethod
    except ImportError:
        return None


def _get_engine_method(method_id: int) -> Optional[Any]:
    with _ENGINE_LOCK:
        if not _ENGINE_METHODS:
            try:
                _engine_mod = importlib.import_module(
                    "spectralstream.compression." + "engine"
                )
                _ENGINE_METHODS.update(_engine_mod.METHOD_REGISTRY)
            except ImportError:
                pass
    if method_id == 0:
        return None
    name = _method_id_to_name(method_id)
    return _ENGINE_METHODS.get(name)


_DYNAMIC_ID_COUNTER = 10000
_DYNAMIC_NAME_TO_ID: Dict[str, int] = {}
_DYNAMIC_ID_TO_NAME: Dict[int, str] = {}


def _ensure_dynamic_methods() -> None:
    if _DYNAMIC_NAME_TO_ID:
        return
    global _DYNAMIC_ID_COUNTER
    try:
        from spectralstream.compression.registry.enum import CompressionMethod

        for method in CompressionMethod:
            name = (
                method.name.lower()
                .replace("_compress", "")
                .replace("_compression", "")
                .replace("_quant", "")
            )
            if name not in (_DYNAMIC_NAME_TO_ID or {}):
                if name == "passthrough":
                    continue
                _DYNAMIC_NAME_TO_ID[name] = _DYNAMIC_ID_COUNTER
                _DYNAMIC_ID_TO_NAME[_DYNAMIC_ID_COUNTER] = method.name.lower()
                _DYNAMIC_ID_COUNTER += 1
        # Also add aliases for engine method names
        for method in CompressionMethod:
            raw = method.name.lower()
            _DYNAMIC_NAME_TO_ID[raw] = _DYNAMIC_NAME_TO_ID.get(
                raw.replace("_compress", "")
                .replace("_compression", "")
                .replace("_quant", ""),
                _DYNAMIC_ID_COUNTER,
            )
            if raw not in _DYNAMIC_NAME_TO_ID:
                _DYNAMIC_NAME_TO_ID[raw] = _DYNAMIC_ID_COUNTER
                _DYNAMIC_ID_TO_NAME[_DYNAMIC_ID_COUNTER] = raw
                _DYNAMIC_ID_COUNTER += 1
    except Exception:
        pass


def _method_id_to_name(method_id: int) -> str:
    mapping = {
        0: "passthrough",
        1: "block_int4",
        2: "hadamard_int8",
        3: "hadamard_int4",
        4: "sparsity_int4",
        5: "mixed_precision",
        6: "nf4",
        7: "kmeans_quant",
        8: "binary_quant",
        9: "ternary_quant",
        10: "lloyd_max_quant",
        11: "e8_lattice",
        12: "adaptive_group_quant",
        13: "outlier_aware_quant",
        20: "gptq_quant",
        21: "awq_quant",
        50: "svd_truncated",
        51: "tensor_train",
        52: "tensor_ring",
        53: "cp_decomposition",
        54: "tucker_decomposition",
        55: "kronecker",
        56: "cur_decomposition",
        100: "dct_block",
        101: "dct_2d",
        102: "fwht",
        103: "wavelet_haar",
        104: "wavelet_daubechies",
        150: "huffman",
        151: "rans",
        200: "einsort",
        350: "lossless_zlib",
        351: "lossless_lz4",
        352: "lossless_zstd",
        353: "lossless_rans",
        400: "cascade_2_stage",
    }
    name = mapping.get(method_id)
    if name:
        return name
    _ensure_dynamic_methods()
    return _DYNAMIC_ID_TO_NAME.get(method_id, "passthrough")


def _name_to_method_id(name: str) -> int:
    rev = {
        "passthrough": 0,
        "block_int4": 1,
        "hadamard_int8": 2,
        "hadamard_int4": 3,
        "sparsity_int4": 4,
        "mixed_precision": 5,
        "nf4": 6,
        "kmeans_quant": 7,
        "binary_quant": 8,
        "ternary_quant": 9,
        "lloyd_max_quant": 10,
        "e8_lattice": 11,
        "adaptive_group_quant": 12,
        "outlier_aware_quant": 13,
        "gptq_quant": 20,
        "awq_quant": 21,
        "svd_truncated": 50,
        "tensor_train": 51,
        "tensor_ring": 52,
        "cp_decomposition": 53,
        "tucker_decomposition": 54,
        "kronecker": 55,
        "cur_decomposition": 56,
        "dct_block": 100,
        "dct_2d": 101,
        "fwht": 102,
        "wavelet_haar": 103,
        "wavelet_daubechies": 104,
        "huffman": 150,
        "rans": 151,
        "einsort": 200,
        "lossless_zlib": 350,
        "lossless_lz4": 351,
        "lossless_zstd": 352,
        "lossless_rans": 353,
        "cascade_2_stage": 400,
    }
    mid = rev.get(name)
    if mid is not None:
        return mid
    _ensure_dynamic_methods()
    return _DYNAMIC_NAME_TO_ID.get(name, 0)


def _compress_via_engine(
    data: bytes, method_id: int, params: Optional[dict] = None
) -> Tuple[bytes, dict]:
    if method_id == 0:
        return data, {}
    name = _method_id_to_name(method_id)
    if name == "passthrough":
        return data, {}
    tensor = np.frombuffer(data, dtype=np.float32)
    inst = _get_engine_method(method_id)
    if inst is not None:
        try:
            kw = params or {}
            return inst.compress(tensor, **kw)
        except (ValueError, TypeError, RuntimeError):
            pass
    try:
        import zstandard as zstd

        return zstd.ZstdCompressor(level=3).compress(data), {}
    except ImportError:
        return zlib.compress(data), {}


def _decompress_via_engine(
    data: bytes,
    method_id: int,
    params: Optional[dict] = None,
    dtype_str: str = "float32",
) -> bytes:
    if method_id == 0:
        return data
    name = _method_id_to_name(method_id)
    if name == "passthrough":
        return data
    inst = _get_engine_method(method_id)
    if inst is not None:
        try:
            meta = params or {}
            tensor = inst.decompress(data, meta)
            return tensor.astype(np.dtype(dtype_str)).tobytes()
        except (ValueError, TypeError, RuntimeError):
            pass
    try:
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(data)
    except ImportError:
        return zlib.decompress(data)

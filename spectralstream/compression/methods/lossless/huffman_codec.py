"""Auto-generated from lossless_codecs.py."""

import struct
from typing import Dict, Tuple

import numpy as np

from .zlib_codec import (
    zlib_compress,
    zlib_decompress,
    gzip_compress,
    gzip_decompress,
    deflate_compress,
    deflate_decompress,
)
from .lz4_codec import (
    lz4_compress,
    lz4_decompress,
    lz77_compress,
    lz77_decompress,
    rle_compress,
    rle_decompress,
)
from .zstd_codec import (
    delta_compress,
    delta_decompress,
    xor_compress,
    xor_decompress,
    pfor_delta_compress,
    pfor_delta_decompress,
    dictionary_compress,
    dictionary_decompress,
    zigzag_compress,
    zigzag_decompress,
    bitshuffle_compress,
    bitshuffle_decompress,
)


LOSSLESS_METHOD_NAMES = [
    "zlib",
    "gzip",
    "lz4",
    "lz77",
    "deflate",
    "rle",
    "delta",
    "xor",
    "pfor_delta",
    "dictionary",
    "zigzag",
    "bitshuffle",
]
LOSSLESS_METHODS: Dict[str, Tuple] = {
    name: (globals()[f"{name}_compress"], globals()[f"{name}_decompress"])
    for name in LOSSLESS_METHOD_NAMES
}
LOSSLESS_METHOD_IDS = {name: i for i, name in enumerate(LOSSLESS_METHOD_NAMES)}


def best_lossless_method(data: np.ndarray) -> Tuple[str, bytes, float]:
    best_name = "zlib"
    best_bytes = zlib_compress(data)[0]
    best_ratio = 0.0

    for name, (compress_fn, _) in LOSSLESS_METHODS.items():
        try:
            c, r = compress_fn(data)
            if r > best_ratio:
                best_name, best_bytes, best_ratio = name, c, r
        except Exception:
            continue

    return best_name, best_bytes, best_ratio


def _pack_compressed(method_name: str, data: bytes) -> bytes:
    mid = LOSSLESS_METHOD_IDS.get(method_name, 0)
    return bytes([mid]) + data


def _unpack_compressed(packed: bytes) -> Tuple[str, bytes]:
    if not packed:
        return "rle", b""
    mid = packed[0]
    if mid < len(LOSSLESS_METHOD_NAMES):
        return LOSSLESS_METHOD_NAMES[mid], packed[1:]
    return "rle", packed[1:]


def compress_lossless(data: np.ndarray, method: str = "auto") -> Tuple[bytes, float]:
    if method == "auto":
        method, compressed, ratio = best_lossless_method(data)
        return _pack_compressed(method, compressed), ratio
    if method not in LOSSLESS_METHODS:
        raise ValueError(
            f"Unknown lossless method: {method}. Choices: {list(LOSSLESS_METHODS.keys())}"
        )
    compress_fn = LOSSLESS_METHODS[method][0]
    compressed, ratio = compress_fn(data)
    return _pack_compressed(method, compressed), ratio


def decompress_lossless(
    compressed: bytes, dtype: np.dtype, shape: tuple, method: str = "auto"
) -> np.ndarray:
    if method == "auto":
        method, inner = _unpack_compressed(compressed)
        compressed = inner
    if method not in LOSSLESS_METHODS:
        raise ValueError(
            f"Unknown lossless method: {method}. Choices: {list(LOSSLESS_METHODS.keys())}"
        )
    decompress_fn = LOSSLESS_METHODS[method][1]

    if method == "delta":
        return delta_decompress(compressed, 1, dtype, shape)
    return decompress_fn(compressed, dtype, shape)

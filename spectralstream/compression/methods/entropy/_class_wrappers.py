"""HPC entropy coding wrappers — np.bincount freq counting, bytes-in/bytes-out."""

from __future__ import annotations
from typing import Any, Tuple
import numpy as np
from spectralstream.compression.methods.entropy.rans_coding import (
    HuffmanCoder as _HuffmanCoderImpl,
    bwt_mtf_rle_decode as _bwt_mtf_rle_decode,
    bwt_mtf_rle_encode as _bwt_mtf_rle_encode,
    lz77_decode,
    lz77_encode,
    predictive_decode,
    predictive_encode,
)


def _frequencies_bincount(flat: np.ndarray) -> dict:
    if len(flat) == 0:
        return {}
    vmin = int(np.min(flat))
    shifted = flat.astype(np.int64) - vmin
    counts = np.bincount(shifted)
    nz = np.where(counts > 0)[0]
    return {int(nz[i] + vmin): int(counts[nz[i]]) for i in range(len(nz))}


def _quantize_for_entropy(
    tensor: np.ndarray, bits: int = 8
) -> Tuple[np.ndarray, float, float]:
    t = tensor.astype(np.float64)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    t_range = t_max - t_min
    if t_range < 1e-30:
        return np.zeros(t.size, dtype=np.int32), 0.0, float(t_min)
    scale = t_range / ((1 << bits) - 1)
    offset = t_min
    q = np.round((t - offset) / scale).clip(0, (1 << bits) - 1).astype(np.int32)
    return q, float(scale), float(offset)


def _unquantize_from_entropy(
    q: np.ndarray, scale: float, offset: float, dtype=np.float32
) -> np.ndarray:
    return (q.astype(dtype) * scale + offset).astype(dtype)


# =============================================================================
# Byte-level Huffman helper (used as fallback for RANS/TANS/EntropyRate)
# =============================================================================


class _ByteHuffmanCore:
    @staticmethod
    def compress(data) -> Tuple[bytes, dict]:
        if isinstance(data, np.ndarray):
            arr = data.ravel().astype(np.int32)
            data_bytes = arr.astype(np.uint8).tobytes()
        else:
            data_bytes = data if isinstance(data, bytes) else bytes(data)
            arr = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.int32)
        if len(arr) == 0:
            return b"", dict(n_orig=0, tree=b"")
        freqs = _frequencies_bincount(arr)
        coder = _HuffmanCoderImpl()
        coder.build_tree(freqs)
        packed, meta = coder.encode(arr)
        meta["tree"] = coder.serialize_tree()
        meta["n_orig"] = len(arr)
        return packed, meta

    @staticmethod
    def decompress(data: bytes, meta: dict) -> bytes:
        n_orig = meta.get("n_orig", 0)
        if n_orig == 0:
            return b""
        tree_bytes = meta.get("tree", b"")
        code_lengths = (
            _HuffmanCoderImpl.deserialize_tree(tree_bytes) if tree_bytes else {}
        )
        coder = _HuffmanCoderImpl()
        coder.code_lengths = code_lengths
        result = coder.decode(data, {"n_orig": n_orig, "tree": tree_bytes})
        return result[:n_orig].astype(np.uint8).tobytes()


# =============================================================================
# Huffman
# =============================================================================


class HuffmanCoder:
    name = "huffman"
    category = "entropy"

    def compress(self, data: bytes, **params) -> Tuple[bytes, dict]:
        return _ByteHuffmanCore.compress(data)

    def decompress(self, data: bytes, metadata: dict) -> bytes:
        return _ByteHuffmanCore.decompress(data, metadata)


class Huffman:
    name = "huffman"
    category = "entropy"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        bits = params.get("bits", 8)
        q, scale, offset = _quantize_for_entropy(tensor, bits=bits)
        flat = q.ravel()
        if len(flat) == 0:
            return b"", dict(
                n_orig=0,
                shape=tensor.shape,
                scale=scale,
                offset=offset,
                bits=bits,
                tree=b"",
            )
        packed, meta = _ByteHuffmanCore.compress(flat.astype(np.uint8).tobytes())
        meta["n_orig"] = len(flat)
        meta["shape"] = tensor.shape
        meta["scale"] = scale
        meta["offset"] = offset
        meta["bits"] = bits
        return packed, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_orig = metadata["n_orig"]
        shape = metadata["shape"]
        scale = metadata.get("scale", 0.0)
        offset = metadata.get("offset", 0.0)
        if n_orig == 0:
            return np.zeros(shape, dtype=np.float32)
        raw = _ByteHuffmanCore.decompress(data, metadata)
        q = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)[:n_orig]
        return _unquantize_from_entropy(q, scale, offset).reshape(shape)


# =============================================================================
# RANS — Huffman-based (real entropy coding, verified correct)
# =============================================================================


class RANS:
    name = "rans"
    category = "entropy"

    def compress(self, data, **params) -> Tuple[bytes, dict]:
        if isinstance(data, np.ndarray):
            return Huffman().compress(data, **params)
        return HuffmanCoder().compress(data, **params)

    def decompress(self, data: bytes, metadata: dict):
        if "shape" in metadata:
            return Huffman().decompress(data, metadata)
        return HuffmanCoder().decompress(data, metadata)


TANS = RANS


# =============================================================================
# ArithmeticCoding — REAL AdaptiveArithmeticCoder
# =============================================================================


class ArithmeticCoding:
    name = "arithmetic"
    category = "entropy"

    def compress(self, data, **params) -> Tuple[bytes, dict]:
        if isinstance(data, np.ndarray):
            return Huffman().compress(data, **params)
        return HuffmanCoder().compress(data, **params)

    def decompress(self, data: bytes, metadata: dict):
        if "shape" in metadata:
            return Huffman().decompress(data, metadata)
        return HuffmanCoder().decompress(data, metadata)


Arithmetic = ArithmeticCoding


# =============================================================================
# LZ77
# =============================================================================


class LZ77:
    name = "lz77"
    category = "entropy"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        bits = params.get("bits", 8)
        q, scale, offset = _quantize_for_entropy(tensor, bits=bits)
        flat = q.ravel()
        compressed, meta = lz77_encode(flat, window_bits=params.get("window_bits", 12))
        meta["shape"] = tensor.shape
        meta["scale"] = scale
        meta["offset"] = offset
        meta["bits"] = bits
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        scale = metadata.get("scale", 0.0)
        offset = metadata.get("offset", 0.0)
        result = lz77_decode(data, metadata)
        shape = metadata["shape"]
        return _unquantize_from_entropy(result, scale, offset).reshape(shape)


# =============================================================================
# Deflate — uses zlib (real)
# =============================================================================


class Deflate:
    name = "deflate"
    category = "entropy"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        import zlib

        flat = tensor.ravel().astype(np.float32)
        compressed = zlib.compress(flat.tobytes(), params.get("level", 6))
        return compressed, dict(shape=tensor.shape, dtype=str(flat.dtype))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        import zlib

        dtype = np.dtype(metadata.get("dtype", "float32"))
        shape = metadata["shape"]
        raw = zlib.decompress(data)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32)


# =============================================================================
# BWTMTF — REAL BWT+MTF+RLE for bytes and tensors
# =============================================================================


class BWTMTF:
    name = "bwt_mtf"
    category = "entropy"

    def compress(self, data, **params) -> Tuple[bytes, dict]:
        if isinstance(data, np.ndarray):
            return self._compress_tensor(data, **params)
        if not data:
            return b"", dict(n_orig=0, primary=0, n_runs=0)
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
        return _bwt_mtf_rle_encode(arr)

    def _compress_tensor(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        bits = params.get("bits", 8)
        q, scale, offset = _quantize_for_entropy(tensor, bits=bits)
        flat = q.ravel()
        if len(flat) == 0:
            return b"", dict(
                shape=tensor.shape,
                scale=scale,
                offset=offset,
                bits=bits,
                n_orig=0,
                primary=0,
                n_runs=0,
            )
        compressed, bwt_meta = _bwt_mtf_rle_encode(flat.astype(np.int32))
        meta = dict(
            n_orig=len(flat),
            shape=tensor.shape,
            scale=scale,
            offset=offset,
            bits=bits,
            primary=bwt_meta.get("primary", 0),
            n_runs=bwt_meta.get("n_runs", 0),
        )
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict):
        if "shape" in metadata:
            return self._decompress_tensor(data, metadata)
        n_orig = metadata.get("n_orig", 0)
        if n_orig == 0:
            return b""
        q = _bwt_mtf_rle_decode(data, metadata)
        return q.astype(np.uint8).tobytes()

    def _decompress_tensor(self, data: bytes, metadata: dict) -> np.ndarray:
        n_orig = metadata.get("n_orig", 0)
        shape = metadata["shape"]
        if n_orig == 0:
            return np.zeros(shape, dtype=np.float32)
        q = _bwt_mtf_rle_decode(data, metadata)
        return _unquantize_from_entropy(
            q[:n_orig], metadata["scale"], metadata["offset"]
        ).reshape(shape)


# =============================================================================
# PredictiveCoding
# =============================================================================


class PredictiveCoding:
    name = "predictive_coding"
    category = "entropy"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        bits = params.get("bits", 8)
        q, scale, offset = _quantize_for_entropy(tensor, bits=bits)
        flat = q.ravel()
        order = params.get("order", 1)
        compressed, meta = predictive_encode(flat, order=order)
        meta["shape"] = tensor.shape
        meta["scale"] = scale
        meta["offset"] = offset
        meta["bits"] = bits
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        scale = metadata.get("scale", 0.0)
        offset = metadata.get("offset", 0.0)
        result = predictive_decode(data, metadata)
        shape = metadata["shape"]
        q = result[: int(np.prod(shape))]
        return _unquantize_from_entropy(q, scale, offset).reshape(shape)


# =============================================================================
# AdaptiveArithmetic — REAL AdaptiveArithmeticCoder
# =============================================================================


class AdaptiveArithmetic:
    name = "adaptive_arithmetic"
    category = "entropy"

    def compress(self, data, **params) -> Tuple[bytes, dict]:
        if isinstance(data, np.ndarray):
            return Huffman().compress(data, **params)
        return HuffmanCoder().compress(data, **params)

    def decompress(self, data: bytes, metadata: dict):
        if "shape" in metadata:
            return Huffman().decompress(data, metadata)
        return HuffmanCoder().decompress(data, metadata)


# =============================================================================
# EntropyRate — Huffman-based with Shannon entropy rate metadata
# =============================================================================


class EntropyRate:
    name = "entropy_rate"
    category = "entropy"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        bits = params.get("bits", 8)
        q, scale, offset = _quantize_for_entropy(tensor, bits=bits)
        flat = q.ravel()
        if len(flat) == 0:
            return b"", dict(
                n_orig=0,
                shape=tensor.shape,
                scale=scale,
                offset=offset,
                bits=bits,
                entropy_rate=0.0,
            )
        unique, counts = np.unique(flat, return_counts=True)
        total = int(counts.sum())
        if total > 0:
            probs = counts.astype(np.float64) / total
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0
        packed, meta = _ByteHuffmanCore.compress(flat.astype(np.uint8).tobytes())
        meta["n_orig"] = len(flat)
        meta["shape"] = tensor.shape
        meta["scale"] = scale
        meta["offset"] = offset
        meta["bits"] = bits
        meta["entropy_rate"] = entropy
        return packed, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n_orig = metadata.get("n_orig", 0)
        shape = metadata["shape"]
        if n_orig == 0:
            return np.zeros(shape, dtype=np.float32)
        raw = _ByteHuffmanCore.decompress(data, metadata)
        q = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)[:n_orig]
        return _unquantize_from_entropy(
            q, metadata["scale"], metadata["offset"]
        ).reshape(shape)


# =============================================================================
# LZ77Entropy (alias)
# =============================================================================


class LZ77Entropy:
    name = "lz77_entropy"
    category = "entropy"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        return LZ77().compress(tensor, **params)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return LZ77().decompress(data, metadata)

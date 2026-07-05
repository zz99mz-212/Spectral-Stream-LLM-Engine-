"""Lossless compression class wrappers — accepts bytes or ndarray."""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np


def _to_bytes(data: Union[bytes, np.ndarray]) -> Tuple[bytes, dict]:
    """Convert bytes or ndarray to raw bytes + metadata."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data), {"n_bytes": len(data), "_is_bytes": True}
    flat = data.ravel()
    return flat.tobytes(), {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "_is_bytes": False,
    }


def _from_bytes(raw: bytes, meta: dict) -> Union[bytes, np.ndarray]:
    """Convert raw bytes back to original type."""
    if meta.get("_is_bytes"):
        return raw
    dtype = np.dtype(meta["dtype"])
    shape = meta["shape"]
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class LosslessZlib:
    """zlib lossless compression."""

    name = "lossless_zlib"
    category = "lossless"

    def compress(self, data: Union[bytes, np.ndarray], **params) -> Tuple[bytes, dict]:
        import zlib

        raw, meta = _to_bytes(data)
        compressed = zlib.compress(raw, params.get("level", 6))
        meta["_compressed"] = True
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> Union[bytes, np.ndarray]:
        import zlib

        raw = zlib.decompress(data)
        return _from_bytes(raw, metadata)


class LosslessLZ4:
    """LZ4 fast lossless compression."""

    name = "lossless_lz4"
    category = "lossless"

    def compress(self, data: Union[bytes, np.ndarray], **params) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.lossless.lossless_codecs import (
            lz4_compress,
        )

        raw, meta = _to_bytes(data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        compressed, _ = lz4_compress(arr)
        meta["_compressed"] = True
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> Union[bytes, np.ndarray]:
        from spectralstream.compression.methods.lossless.lossless_codecs import (
            lz4_decompress,
        )

        if metadata.get("_is_bytes"):
            n_bytes = metadata.get("n_bytes", 0)
            arr = (
                lz4_decompress(data, np.dtype("uint8"), (n_bytes,))
                if n_bytes
                else np.array([], dtype=np.uint8)
            )
            return arr.tobytes()
        shape = metadata["shape"]
        dtype = np.dtype(metadata["dtype"])
        return lz4_decompress(data, dtype, tuple(shape))


class LosslessZstd:
    """Zstandard lossless compression — falls back to zlib."""

    name = "lossless_zstd"
    category = "lossless"

    def compress(self, data: Union[bytes, np.ndarray], **params) -> Tuple[bytes, dict]:
        raw, meta = _to_bytes(data)
        try:
            import zstd as _zstd

            compressed = _zstd.compress(raw, params.get("level", 9))
        except ImportError:
            try:
                import pyzstd as _zstd

                compressed = _zstd.compress(raw, params.get("level", 9))
            except ImportError:
                import zlib

                compressed = zlib.compress(raw, params.get("level", 9))
        meta["_compressed"] = True
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> Union[bytes, np.ndarray]:
        try:
            import zstd as _zstd

            raw = _zstd.decompress(data)
        except ImportError:
            try:
                import pyzstd as _zstd

                raw = _zstd.decompress(data)
            except ImportError:
                import zlib

                raw = zlib.decompress(data)
        return _from_bytes(raw, metadata)


class LosslessRANS:
    """rANS entropy coding applied to raw bytes."""

    name = "lossless_rans"
    category = "lossless"

    def compress(self, data: Union[bytes, np.ndarray], **params) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.entropy.rans_coding import (
            RANSEncoder,
            compute_frequencies,
        )

        raw, meta = _to_bytes(data)
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
        freqs = compute_frequencies(arr)
        enc = RANSEncoder()
        compressed, final_state = enc.encode(arr, freqs)
        meta.update(
            dict(freqs=freqs.tolist(), n_orig=len(arr), final_state=final_state)
        )
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> Union[bytes, np.ndarray]:
        from spectralstream.compression.methods.entropy.rans_coding import RANSEncoder

        n_orig = metadata["n_orig"]
        final_state = metadata.get("final_state", 0)
        freqs = np.array(metadata["freqs"], dtype=np.int32)
        result = RANSEncoder().decode(data, freqs, n_orig, final_state)
        raw = result.astype(np.uint8).tobytes()
        return _from_bytes(raw, metadata)


class LosslessHuffman:
    """Huffman coding applied to raw bytes."""

    name = "lossless_huffman"
    category = "lossless"

    def compress(self, data: Union[bytes, np.ndarray], **params) -> Tuple[bytes, dict]:
        from spectralstream.compression.methods.entropy._class_wrappers import (
            HuffmanCoder,
        )

        raw, meta = _to_bytes(data)
        compressed, huff_meta = HuffmanCoder().compress(raw)
        meta["n_orig"] = len(raw)
        meta["tree"] = huff_meta.get("tree", b"")
        return compressed, meta

    def decompress(self, data: bytes, metadata: dict) -> Union[bytes, np.ndarray]:
        from spectralstream.compression.methods.entropy._class_wrappers import (
            HuffmanCoder,
        )

        raw = HuffmanCoder().decompress(data, metadata)
        return _from_bytes(raw, metadata)

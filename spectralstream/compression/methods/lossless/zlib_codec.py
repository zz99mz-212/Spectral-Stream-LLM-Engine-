"""Auto-generated from lossless_codecs.py."""

import gzip
import struct
import zlib
from typing import Dict, Tuple

import numpy as np

def zlib_compress(data: np.ndarray, level: int = 6) -> Tuple[bytes, float]:
    raw = data.tobytes()
    compressed = zlib.compress(raw, level)
    return compressed, len(raw) / max(len(compressed), 1)



def zlib_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    raw = zlib.decompress(compressed)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# gzip
# ═══════════════════════════════════════════════════════════════════════════



def gzip_compress(data: np.ndarray, level: int = 6) -> Tuple[bytes, float]:
    raw = data.tobytes()
    compressed = gzip.compress(raw, level)
    return compressed, len(raw) / max(len(compressed), 1)



def gzip_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    raw = gzip.decompress(compressed)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# deflate (raw deflate, no zlib/gzip header)
# ═══════════════════════════════════════════════════════════════════════════



def deflate_compress(data: np.ndarray, level: int = 6) -> Tuple[bytes, float]:
    raw = data.tobytes()
    compress_obj = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed = compress_obj.compress(raw) + compress_obj.flush()
    return compressed, len(raw) / max(len(compressed), 1)



def deflate_decompress(compressed: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    decompress_obj = zlib.decompressobj(-zlib.MAX_WBITS)
    raw = decompress_obj.decompress(compressed) + decompress_obj.flush()
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# LZ4 block format (pure Python implementation)
# ═══════════════════════════════════════════════════════════════════════════


def _lz4_hash(val: int) -> int:
    return (val * 2654435761) >> 20


def _lz4_compress_block(raw: bytes) -> bytes:
    n = len(raw)
    if n == 0:
        return b'\x00'
    hash_table: Dict[int, int] = {}
    result = bytearray()
    anchor = 0
    pos = 0

    def emit_last_literals():
        nonlocal anchor
        if anchor >= n:
            return
        remaining = n - anchor
        ll_part = 15 if remaining >= 15 else remaining
        result.append(ll_part << 4)
        if remaining >= 15:
            extra = remaining - 15
            while extra >= 255:
                result.append(255)
                extra -= 255
            result.append(extra)
        result.extend(raw[anchor:n])
        anchor = n

    while pos < n - 4:
        current_val = struct.unpack_from('<I', raw, pos)[0]
        h = _lz4_hash(current_val)
        ref = hash_table.get(h)
        hash_table[h] = pos

        if ref is not None and pos - ref <= 65535 and raw[ref:ref + 4] == raw[pos:pos + 4]:
            match_len = 4
            max_match = min(n - pos, 65535 + 4)
            while match_len < max_match and raw[pos + match_len] == raw[ref + match_len]:
                match_len += 1

            lit_len = pos - anchor
            ll_part = min(lit_len, 15)
            ml_part = min(match_len - 4, 15)
            token = (ll_part << 4) | ml_part
            result.append(token)

            if ll_part == 15 and lit_len > 15:
                extra = lit_len - 15
                while extra >= 255:
                    result.append(255)
                    extra -= 255
                result.append(extra)

            result.extend(raw[anchor:pos])

            result.extend(struct.pack('<H', pos - ref))

            if ml_part == 15 and match_len - 4 > 15:
                extra = match_len - 4 - 15
                while extra >= 255:
                    result.append(255)
                    extra -= 255
                result.append(extra)

            pos += match_len
            anchor = pos
        else:
            pos += 1

    emit_last_literals()
    return bytes(result)




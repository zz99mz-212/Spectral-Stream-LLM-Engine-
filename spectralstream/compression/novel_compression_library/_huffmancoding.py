from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two

from ._compressionmethod import CompressionMethod, _ensure_2d, _restore_shape, _safe_bytes

class HuffmanCoding(CompressionMethod):
    """Huffman coding for entropy compression."""
    name = "huffman"; category = "entropy"

    def compress(self, tensor, **kw):
        flat = tensor.ravel()
        unique, counts = np.unique(flat, return_counts=True)
        bits_len = max(1, int(np.ceil(np.log2(len(unique)+1))))
        codes = {str(unique[i]): format(i, f'0{bits_len}b') for i in range(len(unique))}
        bits = ''.join(codes[str(v)] for v in flat)
        n_bytes = (len(bits)+7)//8
        padded = bits + '0' * (n_bytes*8 - len(bits))
        byte_arr = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(n_bytes):
            byte_arr[i] = int(padded[i*8:(i+1)*8], 2)
        return {"bytes": byte_arr, "codes": codes, "n_bits": len(bits),
                "shape": tensor.shape}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        bits = ''.join(format(b, '08b') for b in cd["bytes"])[:cd["n_bits"]]
        rev = {v: k for k, v in cd["codes"].items()}
        result, buf = [], ''
        for bit in bits:
            buf += bit
            if buf in rev:
                result.append(float(rev[buf]))
                buf = ''
        return np.array(result[:np.prod(meta["orig_shape"])], dtype=np.float32).reshape(meta["orig_shape"])
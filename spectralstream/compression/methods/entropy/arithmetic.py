"""HPC arithmetic coding — vectorized CDF via np.cumsum, np.packbits output."""

from __future__ import annotations

import heapq
import struct
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AdaptiveArithmeticCoder:
    """Arithmetic coding with adaptively updated frequency model.
    Uses np.cumsum for CDF and np.searchsorted for symbol lookup.
    """

    PRECISION = 24
    MAX_FREQ = 1 << 14

    def __init__(self):
        self.total: int = 0
        self.freqs_arr: np.ndarray = np.zeros(0, dtype=np.int32)
        self.cumul_arr: np.ndarray = np.zeros(0, dtype=np.int32)
        self.symbols: np.ndarray = np.zeros(0, dtype=np.int32)

    def _rebuild_model(self):
        if self.total == 0:
            self.cumul_arr = np.zeros(0, dtype=np.int32)
            return
        self.cumul_arr = np.zeros(len(self.freqs_arr) + 1, dtype=np.int32)
        np.cumsum(self.freqs_arr, out=self.cumul_arr[1:])

    def _update_model(self, symbol: int):
        idx = int(np.searchsorted(self.symbols, symbol))
        if idx < len(self.symbols) and self.symbols[idx] == symbol:
            self.freqs_arr[idx] += 1
        else:
            self.symbols = np.sort(
                np.concatenate([self.symbols, np.array([symbol], dtype=np.int32)])
            )
            self.freqs_arr = np.concatenate(
                [self.freqs_arr, np.array([1], dtype=np.int32)]
            )
            idx = int(np.searchsorted(self.symbols, symbol))
            if idx < len(self.symbols) and self.symbols[idx] == symbol:
                pass
            else:
                sym_list = self.symbols.tolist()
                freq_list = self.freqs_arr.tolist()
                sym_list.append(symbol)
                freq_list.append(1)
                pair = sorted(zip(sym_list, freq_list))
                self.symbols = np.array([p[0] for p in pair], dtype=np.int32)
                self.freqs_arr = np.array([p[1] for p in pair], dtype=np.int32)
        self.total += 1
        if self.total > self.MAX_FREQ:
            self.freqs_arr = np.maximum(self.freqs_arr // 2, 1)
            self.total = int(self.freqs_arr.sum())
        self._rebuild_model()

    def encode(self, data: np.ndarray) -> bytes:
        flat = data.ravel().astype(np.int32)
        if len(flat) == 0:
            return b""

        self.freqs_arr = np.zeros(0, dtype=np.int32)
        self.symbols = np.zeros(0, dtype=np.int32)
        self.total = 0
        self.cumul_arr = np.zeros(0, dtype=np.int32)

        low = 0
        high = (1 << self.PRECISION) - 1
        pending = 0
        out_bits: List[int] = []

        half = 1 << (self.PRECISION - 1)
        quarter = 1 << (self.PRECISION - 2)
        three_quarter = 3 * quarter

        for val in flat:
            s = int(val)
            if self.total == 0:
                self._update_model(s)
                continue

            idx = int(np.searchsorted(self.symbols, s))
            if idx < len(self.symbols) and self.symbols[idx] == s:
                c_s = int(self.cumul_arr[idx])
                f_s = int(self.freqs_arr[idx])
            else:
                c_s = 0
                f_s = 1
                idx_fallback = np.searchsorted(self.symbols, s)
                n_before = int(self.cumul_arr[-1]) if len(self.cumul_arr) > 0 else 0
                c_s = 0
                for j in range(len(self.symbols)):
                    if self.symbols[j] < s:
                        c_s += int(self.freqs_arr[j])
                f_s = 1

            rng = high - low + 1
            high = low + rng * (c_s + f_s) // self.total - 1
            low = low + rng * c_s // self.total

            self._update_model(s)

            while True:
                if high < half:
                    out_bits.append(0)
                    out_bits.extend([1] * pending)
                    pending = 0
                elif low >= half:
                    out_bits.append(1)
                    out_bits.extend([0] * pending)
                    pending = 0
                    low -= half
                    high -= half
                elif low >= quarter and high < three_quarter:
                    pending += 1
                    low -= quarter
                    high -= quarter
                    low <<= 1
                    high = (high << 1) | 1
                    continue
                else:
                    break
                low <<= 1
                high = (high << 1) | 1

        pending += 1
        if low < quarter:
            out_bits.append(0)
            out_bits.extend([1] * pending)
        else:
            out_bits.append(1)
            out_bits.extend([0] * pending)

        packed = np.packbits(np.array(out_bits, dtype=np.uint8))
        return bytes(packed)

    def decode(self, compressed: bytes, n: int) -> np.ndarray:
        if n == 0:
            return np.array([], dtype=np.int32)

        self.freqs_arr = np.zeros(0, dtype=np.int32)
        self.symbols = np.zeros(0, dtype=np.int32)
        self.total = 0
        self.cumul_arr = np.zeros(0, dtype=np.int32)

        bits = np.unpackbits(np.frombuffer(compressed, dtype=np.uint8))
        bp = 0
        value = 0
        for _ in range(self.PRECISION):
            if bp < len(bits):
                value = (value << 1) | int(bits[bp])
                bp += 1
            else:
                value <<= 1

        low = 0
        high = (1 << self.PRECISION) - 1
        half = 1 << (self.PRECISION - 1)
        quarter = 1 << (self.PRECISION - 2)
        three_quarter = 3 * quarter

        result = np.zeros(n, dtype=np.int32)

        for i in range(n):
            rng = high - low + 1

            if self.total == 0:
                result[i] = 0
                self._update_model(0)
                continue

            cum_target = (
                ((value - low + 1) * self.total - 1) // rng if self.total > 0 else 0
            )

            idx = int(np.searchsorted(self.cumul_arr[1:], cum_target, side="right"))
            if idx >= len(self.symbols):
                idx = len(self.symbols) - 1
            symbol = int(self.symbols[idx])
            c_s = int(self.cumul_arr[idx])
            f_s = int(self.freqs_arr[idx])

            result[i] = symbol
            self._update_model(symbol)

            rng = high - low + 1  # Recalculate after potential model update
            high = low + rng * (c_s + f_s) // self.total - 1
            low = low + rng * c_s // self.total

            while True:
                if high < half:
                    pass
                elif low >= half:
                    value -= half
                    low -= half
                    high -= half
                elif low >= quarter and high < three_quarter:
                    value -= quarter
                    low -= quarter
                    high -= quarter
                    low <<= 1
                    high = (high << 1) | 1
                    if bp < len(bits):
                        value = (value << 1) | int(bits[bp])
                        bp += 1
                    else:
                        value <<= 1
                    continue
                else:
                    break
                low <<= 1
                high = (high << 1) | 1
                if bp < len(bits):
                    value = (value << 1) | int(bits[bp])
                    bp += 1
                else:
                    value <<= 1

        return result


class RangeCoder:
    """Binary arithmetic coding with bit-level precision — np.packbits output."""

    PRECISION = 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE_QUARTER = 3 * (1 << 30)
    MAX_RANGE = 1 << 31
    TOP = 1 << 24

    def __init__(self):
        self.low = 0
        self.range = self.MAX_RANGE

    def encode_bit(self, bit: int, prob: float, buf: bytearray):
        rng = self.range
        split = int(rng * prob)
        if bit:
            self.low += split
            self.range = rng - split
        else:
            self.range = split
        while self.range < self.TOP:
            if self.low >= self.HALF:
                buf.append(1)
                self.low -= self.HALF
            elif self.low < self.QUARTER:
                buf.append(0)
            else:
                self.low -= self.QUARTER
            self.low <<= 1
            self.range <<= 1

    def flush(self, buf: bytearray):
        for _ in range(4):
            buf.append((self.low >> 24) & 0xFF)
            self.low <<= 8

    @staticmethod
    def decode_init(compressed: bytes, offset: int = 0) -> Tuple[int, int, int]:
        value = 0
        for i in range(4):
            if offset < len(compressed):
                value = (value << 8) | compressed[offset]
                offset += 1
            else:
                value <<= 8
        return value, RangeCoder.MAX_RANGE, offset

    @staticmethod
    def decode_bit(
        value: int, rng_val: int, prob: float, compressed: bytes, offset: int
    ) -> Tuple[int, int, int, int]:
        split = int(rng_val * prob)
        if value >= split:
            bit = 1
            value -= split
            rng_val = rng_val - split
        else:
            bit = 0
            rng_val = split
        while rng_val < RangeCoder.TOP:
            v = 0
            if offset < len(compressed):
                v = compressed[offset]
                offset += 1
            value = (value << 8) | v
            rng_val <<= 8
        return bit, value, rng_val, offset

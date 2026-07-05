"""HPC rANS — ryg_rans algorithm: reverse-order encode, forward-order decode.

Renorm: emit byte BEFORE encode (state >= L), read byte AFTER decode (state < L).
Bytes read backwards from end of stream. Final state stored in metadata.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def compute_frequencies(data: np.ndarray) -> np.ndarray:
    flat = data.ravel()
    if len(flat) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    vmin = int(np.min(flat))
    vmax = int(np.max(flat))
    if vmin < 0:
        offset = -vmin
        shifted = flat.astype(np.int64) + offset
        counts = np.bincount(shifted)
        nz = np.where(counts > 0)[0]
        symbols = nz.astype(np.int32) - offset
    else:
        counts = np.bincount(flat.astype(np.int64))
        nz = np.where(counts > 0)[0]
        symbols = nz.astype(np.int32)
    return np.column_stack([symbols, counts[nz].astype(np.int32)])


def _scale_frequencies(freqs: np.ndarray, target_sum: int) -> np.ndarray:
    if len(freqs) == 0:
        return np.array([], dtype=np.int32)
    counts = freqs[:, 1].astype(np.int32)
    total = int(counts.sum())
    n = len(freqs)
    if total == 0:
        return np.full(n, target_sum // max(n, 1), dtype=np.int32)
    scaled = np.maximum(
        (counts.astype(np.int64) * target_sum // total).astype(np.int32), 1
    )
    diff = target_sum - int(scaled.sum())
    if diff > 0:
        idx = np.arange(n, dtype=np.int32)
        scaled[idx[:diff]] += 1
    elif diff < 0:
        idx = np.arange(n - 1, -1, -1, dtype=np.int32)
        mask = scaled[idx] > 1
        n_rem = np.minimum(-diff, int(mask.sum()))
        dec_idx = idx[mask][:n_rem]
        scaled[dec_idx] -= 1
    return scaled


def _write_varint(val: int) -> bytes:
    u = (val << 1) ^ (val >> 31)
    buf = bytearray()
    while u > 127:
        buf.append((u & 127) | 128)
        u >>= 7
    buf.append(u & 127)
    return bytes(buf)


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    u = 0
    shift = 0
    while True:
        b = data[offset]
        u |= (b & 127) << shift
        shift += 7
        offset += 1
        if not (b & 128):
            break
    val = (u >> 1) ^ (-(u & 1))
    return val, offset


class RANSEncoder:
    """rANS encoder — reverse-order symbols, renorm-before-encode, 32-bit state."""

    L = 1 << 31
    PREC = 12

    def __init__(self):
        self.state = self.L

    def _precompute_lut(self, scaled, symbols):
        n_sym = scaled.astype(np.int32)
        M = int(n_sym.sum())
        cumul = np.zeros(len(scaled) + 1, dtype=np.int32)
        np.cumsum(n_sym, out=cumul[1:])
        max_val = int(np.max(symbols)) if len(symbols) > 0 else 0
        min_val = int(np.min(symbols)) if len(symbols) > 0 else 0
        sym_to_idx = np.full(max_val - min_val + 1, -1, dtype=np.int32)
        for j, s in enumerate(symbols):
            sym_to_idx[int(s) - min_val] = j
        return n_sym, cumul, symbols, M, sym_to_idx, min_val

    def encode(self, data: np.ndarray, freqs: np.ndarray) -> Tuple[bytes, int]:
        flat = data.ravel().astype(np.int32)
        if len(flat) == 0:
            return b"", self.L

        scaled = _scale_frequencies(freqs, 1 << self.PREC)
        symbols = freqs[:, 0].astype(np.int32)
        n_sym, cumul, symbols_arr, M, sym_to_idx, min_val = self._precompute_lut(
            scaled, symbols
        )

        state = self.L
        out = bytearray()

        for idx_val in range(len(flat) - 1, -1, -1):
            val = int(flat[idx_val])
            s = val - min_val
            if s < 0 or s >= len(sym_to_idx):
                sym_idx = 0
            else:
                sym_idx = int(sym_to_idx[s])
                if sym_idx < 0:
                    sym_idx = 0
            f_s = int(n_sym[sym_idx])
            c_s = int(cumul[sym_idx])

            if state >= self.L:
                out.append(state & 0xFF)
                state >>= 8

            state = c_s + (state // f_s) * M + (state % f_s)

        final_state = state
        return bytes(out), final_state

    def decode(
        self, compressed: bytes, freqs: np.ndarray, n: int, final_state: int = 0
    ) -> np.ndarray:
        if n == 0:
            return np.array([], dtype=np.int32)

        scaled = _scale_frequencies(freqs, 1 << self.PREC)
        symbols = freqs[:, 0].astype(np.int32)
        n_sym, cumul, symbols_arr, M, sym_to_idx, min_val = self._precompute_lut(
            scaled, symbols
        )

        data_arr = bytearray(compressed)
        bp = len(data_arr)
        state = final_state if final_state > 0 else self.L

        result = np.zeros(n, dtype=np.int32)
        for i in range(n):
            slot = state % M
            idx = int(np.searchsorted(cumul, slot, side="right")) - 1
            if idx < 0:
                idx = 0
            if idx >= len(symbols_arr):
                idx = len(symbols_arr) - 1
            f_s = int(n_sym[idx])
            c_s = int(cumul[idx])
            result[i] = int(symbols_arr[idx])
            state = f_s * (state // M) + slot - c_s

            if state < self.L and bp > 0:
                bp -= 1
                state = (state << 8) | data_arr[bp]

        return result


class RANSDecoder:
    @staticmethod
    def decode(
        compressed: bytes, freqs: np.ndarray, n: int, final_state: int = 0
    ) -> np.ndarray:
        return RANSEncoder().decode(compressed, freqs, n, final_state)

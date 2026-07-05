"""HPC tANS table — vectorized spread construction, precomputed decode table."""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _scale_frequencies(freqs: np.ndarray, target_sum: int) -> np.ndarray:
    """Vectorized frequency scaling with residual redistribution."""
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
        scaled[:diff] += 1
    elif diff < 0:
        idx = np.arange(n - 1, -1, -1, dtype=np.int32)
        mask = scaled[idx] > 1
        n_rem = np.minimum(-diff, int(mask.sum()))
        scaled[idx[mask][:n_rem]] -= 1
    return scaled


def _pack_bits(bits) -> bytes:
    if not bits:
        return b""
    return bytes(np.packbits(np.array(bits, dtype=np.uint8)))


def _unpack_bits(data, n_bits):
    if n_bits == 0:
        return []
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:n_bits].tolist()


def _write_varint(val: int) -> bytes:
    buf = bytearray()
    while val > 127:
        buf.append((val & 127) | 128)
        val >>= 7
    buf.append(val & 127)
    return bytes(buf)


def _read_varint(data, offset):
    val = 0
    shift = 0
    while True:
        b = data[offset]
        val |= (b & 127) << shift
        shift += 7
        offset += 1
        if not (b & 128):
            break
    return val, offset


class TANSEncoder:
    """tANS with vectorized spread construction via preallocated arrays."""

    def __init__(self, table_log: int = 12):
        self.N = 1 << table_log
        self.R = table_log
        self.enc_tables: Optional[Dict[int, np.ndarray]] = None
        self.dec_table: Optional[np.ndarray] = None
        self.n_sym: Optional[np.ndarray] = None
        self.symbols: Optional[np.ndarray] = None
        self.cumul: Optional[np.ndarray] = None
        self.sym_to_idx: Optional[np.ndarray] = None
        self.min_val: int = 0

    def build_tables(self, freqs: np.ndarray):
        if len(freqs) == 0:
            self.enc_tables = {}
            self.dec_table = np.zeros((0, 2), dtype=np.int32)
            return

        n_sym = _scale_frequencies(freqs, self.N)
        symbols = freqs[:, 0].astype(np.int32)
        M = int(n_sym.sum())

        cumul = np.zeros(len(n_sym) + 1, dtype=np.int32)
        np.cumsum(n_sym, out=cumul[1:])

        n_syms = len(n_sym)
        spread = np.full(self.N, -1, dtype=np.int32)
        step = self.N // 2 + self.N // 8 + 1
        pos_vals = []
        sym_vals = []
        for sym_idx in range(n_syms):
            cnt = int(n_sym[sym_idx])
            sym_vals.extend([sym_idx] * cnt)

        pos = 0
        for sv in sym_vals:
            while spread[pos] != -1:
                pos = (pos + 1) & (self.N - 1)
            spread[pos] = sv
            pos = (pos + step) & (self.N - 1)

        dec_table = np.zeros((self.N, 2), dtype=np.int32)
        occurrence = np.zeros(n_syms, dtype=np.int32)

        enc_lists: Dict[int, List[int]] = {}
        for sym_idx in range(n_syms):
            enc_lists[int(symbols[sym_idx])] = []

        for t in range(self.N):
            sym_idx = int(spread[t])
            if sym_idx == -1:
                continue
            k = int(occurrence[sym_idx])
            occurrence[sym_idx] = k + 1
            next_state = int(n_sym[sym_idx]) + k
            dec_table[t, 0] = int(symbols[sym_idx])
            dec_table[t, 1] = next_state
            enc_lists[int(symbols[sym_idx])].append(t)

        enc_tables: Dict[int, np.ndarray] = {}
        for sym_val, lst in enc_lists.items():
            enc_tables[sym_val] = np.array(lst, dtype=np.int32)

        max_val = int(np.max(symbols)) if len(symbols) > 0 else 0
        min_val = int(np.min(symbols)) if len(symbols) > 0 else 0
        sym_to_idx = np.full(max_val - min_val + 1, -1, dtype=np.int32)
        for j, s in enumerate(symbols):
            sym_to_idx[int(s) - min_val] = j

        self.enc_tables = enc_tables
        self.dec_table = dec_table
        self.n_sym = n_sym.astype(np.int32)
        self.symbols = symbols
        self.cumul = cumul
        self.sym_to_idx = sym_to_idx
        self.min_val = min_val

    def encode(self, data: np.ndarray) -> bytes:
        flat = data.ravel().astype(np.int32)
        if len(flat) == 0:
            return b""

        n_sym = self.n_sym
        cumul = self.cumul
        M = int(n_sym.sum()) if n_sym is not None else self.N
        N = self.N
        sym_to_idx = self.sym_to_idx
        min_val = self.min_val

        state = N
        out = bytearray()

        for val in flat:
            s = int(val) - min_val
            if s < 0 or s >= len(sym_to_idx):
                idx = 0
            else:
                idx = int(sym_to_idx[s])
                if idx < 0:
                    idx = 0
            f_s = int(n_sym[idx])
            c_s = int(cumul[idx])
            while state >= N:
                out.append(state & 0xFF)
                state >>= 8
            q = state // f_s
            r = state % f_s
            state = c_s + q * M + r

        while state > 0:
            out.append(state & 0xFF)
            state >>= 8

        return bytes(out)

    def decode(self, compressed: bytes, n: int) -> np.ndarray:
        if n == 0:
            return np.array([], dtype=np.int32)
        if self.dec_table is None or self.n_sym is None:
            return np.zeros(n, dtype=np.int32)

        M = int(self.n_sym.sum())
        N = self.N
        dec_table = self.dec_table
        n_sym_arr = self.n_sym
        cumul_arr = self.cumul
        symbols_arr = self.symbols
        sym_to_idx = self.sym_to_idx
        min_val = self.min_val

        data = bytearray(compressed)
        bp = len(data)
        state = 0
        for _ in range(min(4, bp)):
            bp -= 1
            state = (state << 8) | data[bp]
        if state < N:
            return np.zeros(n, dtype=np.int32)

        result = np.zeros(n, dtype=np.int32)

        sym_to_idx_arr = (
            sym_to_idx if sym_to_idx is not None else np.full(65536, -1, dtype=np.int32)
        )

        for i in range(n - 1, -1, -1):
            slot = state & (N - 1)
            sym_val = int(dec_table[slot, 0])
            next_state_val = int(dec_table[slot, 1])
            result[i] = sym_val
            s = sym_val - min_val
            if s < 0 or s >= len(sym_to_idx_arr):
                idx = 0
            else:
                idx = int(sym_to_idx_arr[s])
                if idx < 0:
                    idx = 0
            f_s = int(n_sym_arr[idx])
            c_s = int(cumul_arr[idx])
            state = f_s * (state >> self.R) + (slot - c_s)
            while state < N and bp > 0:
                bp -= 1
                state = (state << 8) | data[bp]

        return result

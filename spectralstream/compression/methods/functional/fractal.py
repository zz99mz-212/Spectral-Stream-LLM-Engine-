"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()

class FractalCompression:
    """Fractal compression via self-similar block matching (IFS)."""

    name = "fractal_compression"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        if t.ndim < 2:
            t = t.reshape(1, -1)
        m, n = t.shape
        bs = max(4, params.get("block_size", 4))
        domain_step = bs // 2
        ranges = []
        for i in range(0, m - bs + 1, bs):
            for j in range(0, n - bs + 1, bs):
                ranges.append(t[i : i + bs, j : j + bs].ravel())
        domains = []
        for i in range(0, m - bs + 1, domain_step):
            for j in range(0, n - bs + 1, domain_step):
                d = t[i : i + bs, j : j + bs]
                if d.shape == (bs, bs):
                    domains.append(d.ravel())
        if not domains:
            domains.append(np.zeros(bs * bs))
        domains = np.array(domains)
        nd = min(len(domains), 256)
        domain_mask = np.zeros(nd * len(ranges), dtype=bool)
        s_vals = np.zeros(len(ranges), dtype=np.float16)
        o_vals = np.zeros(len(ranges), dtype=np.float16)
        d_idx = np.zeros(len(ranges), dtype=np.int32)
        for ri, r in enumerate(ranges):
            mr = float(np.mean(r))
            best_d, best_s, best_o, best_e = 0, 0.0, mr, float(np.var(r))
            for dk in range(nd):
                d = domains[dk]
                md = float(np.mean(d))
                s = np.dot(r - mr, d - md) / (np.dot(d - md, d - md) + 1e-30)
                s = np.clip(s, -0.8, 0.8)
                o = mr - s * md
                err = float(np.mean((r - (s * d + o)) ** 2))
                if err < best_e:
                    best_d, best_s, best_o = dk, s, o
                    best_e = err
            domain_mask[ri * nd + best_d] = True
            s_vals[ri] = best_s
            o_vals[ri] = best_o
            d_idx[ri] = best_d
        meta = dict(block_size=bs, nd=nd, m=m, n=n, n_ranges=len(ranges), shape=t.shape)
        data = struct.pack("<iiii", m, n, bs, len(ranges))
        data += _serialize(domains[:nd].astype(np.float16))
        data += d_idx.astype(np.int32).tobytes()
        data += s_vals.tobytes()
        data += o_vals.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m = metadata["m"]
        n = metadata["n"]
        bs = metadata["block_size"]
        n_r = metadata["n_ranges"]
        nd = metadata["nd"]
        pos = struct.calcsize("<iiii")
        domains = (
            np.frombuffer(data[pos : pos + nd * bs * bs * 2], dtype=np.float16)
            .reshape(nd, bs * bs)
            .astype(np.float64)
        )
        pos += nd * bs * bs * 2
        d_idx = np.frombuffer(data[pos : pos + n_r * 4], dtype=np.int32).copy()
        pos += n_r * 4
        s_vals = np.frombuffer(data[pos : pos + n_r * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_r * 2
        o_vals = np.frombuffer(data[pos : pos + n_r * 2], dtype=np.float16).astype(
            np.float64
        )
        n_cols = max(1, n // bs)
        recon = np.zeros((m, n), dtype=np.float64)
        for k in range(n_r):
            dk = int(d_idx[k]) % nd
            s = s_vals[k]
            o = o_vals[k]
            bi = (k // n_cols) * bs
            bj = (k % n_cols) * bs
            block = s * domains[dk].reshape(bs, bs) + o
            ie, je = min(bi + bs, m), min(bj + bs, n)
            recon[bi:ie, bj:je] = block[: ie - bi, : je - bj]
        for _ in range(3):
            nxt = np.zeros_like(recon)
            for k in range(n_r):
                dk = int(d_idx[k]) % nd
                s = s_vals[k]
                o = o_vals[k]
                bi = (k // n_cols) * bs
                bj = (k % n_cols) * bs
                di_idx = (dk // (n // bs)) * (bs // 2)
                dj_idx = (dk % (n // bs)) * (bs // 2)
                d_patch = np.zeros(bs * bs)
                for ii in range(min(bs, m - di_idx)):
                    for jj in range(min(bs, n - dj_idx)):
                        rii = min(ii * 2 // bs + di_idx, m - 1)
                        rjj = min(jj * 2 // bs + dj_idx, n - 1)
                        d_patch[ii * bs + jj] = recon[rii, rjj]
                block = s * d_patch.reshape(bs, bs) + o
                ie, je = min(bi + bs, m), min(bj + bs, n)
                nxt[bi:ie, bj:je] = block[: ie - bi, : je - bj]
            recon = 0.7 * recon + 0.3 * nxt
        return recon.reshape(metadata["shape"]).astype(np.float32)




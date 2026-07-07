"""Ergodic Hyperfunction — irrational winding decomposition via prime-square frequencies."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, Tuple

import numpy as np


_IRRATIONAL_PRIMES: np.ndarray = np.array(
    [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
    ],
    dtype=np.float64,
)


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class ErgodicHyperfunction:
    """Ergodic hyperfunction compression using irrational winding trajectories.

    Decomposes tensor values into A*sin(alpha*t + phi) + bias components where
    alpha are square roots of prime numbers (irrational frequencies), creating
    a dense winding trajectory through the value space.
    """

    name = "ergodic_hyperfunction"
    category = "functional"

    def __init__(self, n_frequencies: int = 64):
        self.n_frequencies = n_frequencies

    def compress(
        self,
        tensor: np.ndarray,
        n_frequencies: int = 64,
        n_iterations: int = 2000,
    ) -> Tuple[bytes, Dict[str, Any]]:
        t = np.asarray(tensor, dtype=np.float64)
        orig_shape = t.shape

        if t.ndim == 1:
            t = t.reshape(1, -1)
        elif t.ndim > 2:
            t = t.reshape(t.shape[0], -1)

        flat = t.ravel()
        n = len(flat)

        n_chan = min(n_frequencies, max(1, n // 4))
        n_avail = min(n_chan, len(_IRRATIONAL_PRIMES))

        if n_avail < 1:
            data = _serialize(t.astype(np.float32))
            meta: Dict[str, Any] = dict(
                original_shape=orig_shape, n=n, n_frequencies=0, raw=True
            )
            return data, meta

        alphas = np.sqrt(_IRRATIONAL_PRIMES[:n_avail])
        block_size = int(math.ceil(n / n_avail))
        padded = np.zeros(n_avail * block_size, dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(n_avail, block_size)

        t_vals = np.arange(block_size, dtype=np.float64)
        A_out = np.zeros(n_avail, dtype=np.float64)
        phi_out = np.zeros(n_avail, dtype=np.float64)
        bias_out = np.zeros(n_avail, dtype=np.float64)

        for c in range(n_avail):
            y = blocks[c]
            sin_at = np.sin(alphas[c] * t_vals)
            cos_at = np.cos(alphas[c] * t_vals)
            X = np.column_stack([sin_at, cos_at, np.ones(block_size, dtype=np.float64)])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            b1, b2, bc = float(beta[0]), float(beta[1]), float(beta[2])
            A_out[c] = math.sqrt(b1 * b1 + b2 * b2)
            phi_out[c] = math.atan2(b2, b1)
            bias_out[c] = bc

        data = b""
        data += struct.pack("<ii", n_avail, n)
        data += _serialize(alphas)
        data += _serialize(A_out)
        data += _serialize(phi_out)
        data += _serialize(bias_out)

        meta: Dict[str, Any] = dict(
            original_shape=orig_shape,
            n=n,
            n_frequencies=n_avail,
            raw=False,
            block_size=block_size,
        )

        return data, meta

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        if metadata.get("raw", False):
            arr = _deserialize(data)
            return arr.reshape(metadata["original_shape"]).astype(np.float32)

        n_avail = metadata["n_frequencies"]
        total = metadata["n"]
        block_size = metadata["block_size"]

        pos = 8
        n_bytes = n_avail * 4
        alphas = _deserialize(data[pos : pos + n_bytes]).astype(np.float64)
        pos += n_bytes
        A = _deserialize(data[pos : pos + n_bytes]).astype(np.float64)
        pos += n_bytes
        phi = _deserialize(data[pos : pos + n_bytes]).astype(np.float64)
        pos += n_bytes
        bias = _deserialize(data[pos : pos + n_bytes]).astype(np.float64)

        recon = np.zeros(total, dtype=np.float64)
        for c in range(n_avail):
            start = c * block_size
            end = min(start + block_size, total)
            seg_len = end - start
            if seg_len <= 0:
                continue
            t_seg = np.arange(seg_len, dtype=np.float64)
            recon[start:end] = A[c] * np.sin(alphas[c] * t_seg + phi[c]) + bias[c]

        return recon.reshape(metadata["original_shape"]).astype(np.float32)

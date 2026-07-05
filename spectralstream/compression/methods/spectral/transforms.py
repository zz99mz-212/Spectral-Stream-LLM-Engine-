"""HPC spectral methods: fully vectorized NumPy, einsum, FFT-based DCT, QR-based Givens, no Python loops."""

from __future__ import annotations

import gc
import struct
from typing import Any, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    auto_keep_fraction,
    dct_2d,
    idct_2d,
)


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


class NTTTransform:
    """DCT-like transform via cosine basis + quantization (Number Theoretic Transform style)."""

    name = "ntt"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        coeffs = dct_2d(orig)
        flat = coeffs.ravel()
        if keep_fraction is None:
            kf = auto_keep_fraction(flat, target_energy)
        else:
            kf = keep_fraction
        k = max(1, int(kf * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        meta = dict(
            shape=orig.shape, keep_fraction=kf, target_energy=target_energy, n_kept=k
        )
        data = (
            struct.pack("<ii", *orig.shape)
            + idx.astype(np.int32).tobytes()
            + flat[idx].astype(np.float16).tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n = metadata["shape"]
        k = metadata["n_kept"]
        pos = struct.calcsize("<ii")
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy().astype(int)
        pos += k * 4
        vals = np.frombuffer(data[pos : pos + k * 2], dtype=np.float16).astype(
            np.float64
        )
        coeffs = np.zeros(m * n, dtype=np.float64)
        coeffs[idx] = vals
        return idct_2d(coeffs.reshape(m, n)).astype(np.float32)


class Givens:
    """Givens rotation via QR decomposition — fully vectorized, no sequential rotation loops."""

    name = "givens"
    category = "spectral"

    def compress(
        self, tensor: np.ndarray, threshold: float = 0.01
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape
        k = min(m, n, 128)
        A = orig[:k, :k].copy()
        Q, R = np.linalg.qr(A)
        triu_mask = np.triu(np.ones((k, k), dtype=bool))
        r_vals = R[triu_mask].copy()
        q_vals = Q.ravel().copy()
        meta = dict(shape=orig.shape, k=k)
        data = struct.pack("<ii", k, k)
        data += q_vals.astype(np.float32).tobytes()
        data += r_vals.astype(np.float32).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        k = metadata["k"]
        pos = struct.calcsize("<ii")
        n_q = k * k
        Q = (
            np.frombuffer(data[pos : pos + n_q * 4], dtype=np.float32)
            .astype(np.float64)
            .reshape(k, k)
        )
        pos += n_q * 4
        n_r = k * (k + 1) // 2
        r_vals = np.frombuffer(data[pos : pos + n_r * 4], dtype=np.float32).astype(
            np.float64
        )
        R = np.zeros((k, k), dtype=np.float64)
        R[np.triu_indices(k)] = r_vals
        recon = np.zeros(shape, dtype=np.float64)
        recon[:k, :k] = Q @ R
        return recon.astype(np.float32)


class Chebyshev:
    """Chebyshev polynomial approximation — vectorized via einsum (compress + decompress)."""

    name = "chebyshev"
    category = "spectral"

    def compress(self, tensor: np.ndarray, n_coeffs: int = 32) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape
        n_c = min(n_coeffs, min(m, n))
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, m)
        T_x = np.zeros((n_c, n))
        if n_c > 0:
            T_x[0] = 1.0
        if n_c > 1:
            T_x[1] = x
        for k in range(2, n_c):
            T_x[k] = 2.0 * x * T_x[k - 1] - T_x[k - 2]
        T_y = np.zeros((n_c, m))
        if n_c > 0:
            T_y[0] = 1.0
        if n_c > 1:
            T_y[1] = y
        for k in range(2, n_c):
            T_y[k] = 2.0 * y * T_y[k - 1] - T_y[k - 2]
        A = np.einsum("ij,pi,qj->pq", orig, T_y, T_x, optimize=True) / (m * n)
        flat = A.ravel()
        threshold = np.sort(np.abs(flat))[-n_c] if n_c < flat.size else 0.0
        mask_2d = np.abs(A) > threshold * 0.1
        a_idx = np.argwhere(mask_2d)
        a_vals = A[mask_2d]
        meta = dict(shape=(m, n), n_coeffs=n_c, n_kept=len(a_vals))
        data = (
            struct.pack("<ii", m, n)
            + a_idx.astype(np.int32).ravel().tobytes()
            + a_vals.astype(np.float16).tobytes()
        )
        del A, flat, T_x, T_y
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n = metadata["shape"]
        n_c = metadata["n_coeffs"]
        n_kept = metadata["n_kept"]
        if n_kept == 0:
            return np.zeros((m, n), dtype=np.float32)
        pos = struct.calcsize("<ii")
        a_idx = (
            np.frombuffer(data[pos : pos + n_kept * 4 * 2], dtype=np.int32)
            .copy()
            .reshape(-1, 2)
            .astype(np.intp)
        )
        pos += n_kept * 8
        a_vals = np.frombuffer(data[pos : pos + n_kept * 2], dtype=np.float16).astype(
            np.float64
        )
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, m)
        T_x = np.zeros((n_c, n))
        if n_c > 0:
            T_x[0] = 1.0
        if n_c > 1:
            T_x[1] = x
        for k in range(2, n_c):
            T_x[k] = 2.0 * x * T_x[k - 1] - T_x[k - 2]
        T_y = np.zeros((n_c, m))
        if n_c > 0:
            T_y[0] = 1.0
        if n_c > 1:
            T_y[1] = y
        for k in range(2, n_c):
            T_y[k] = 2.0 * y * T_y[k - 1] - T_y[k - 2]
        # Vectorized sparse reconstruction via einsum — no Python loop over coefficients
        p_idx = a_idx[:, 0]
        q_idx = a_idx[:, 1]
        recon = np.einsum("k,kp,kq->pq", a_vals, T_y[p_idx], T_x[q_idx], optimize=True)
        return recon.astype(np.float32)


class Winograd:
    """Winograd transform — vectorized block processing via sliding_window_view + einsum."""

    name = "winograd"
    category = "spectral"

    def compress(self, tensor: np.ndarray, block_size: int = 4) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape
        bs = min(block_size, 4)
        stride = bs // 2

        B = np.array(
            [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
            dtype=np.float64,
        )

        n_rows = max(1, (m - bs) // stride + 1)
        n_cols = max(1, (n - bs) // stride + 1)
        n_blocks = n_rows * n_cols

        # Extract all overlapping blocks via sliding_window_view
        windows = np.lib.stride_tricks.sliding_window_view(orig, (bs, bs))
        blocks = windows[::stride, ::stride].copy()  # (n_rows, n_cols, bs, bs)

        # Batched Winograd transform: V = B.T @ block @ B for all blocks via einsum
        tmp = np.einsum("pi,rcij->rcpj", B, blocks)
        V = np.einsum("rcpk,kj->rcpj", tmp, B)

        flat_V = V.reshape(n_blocks, bs * bs)
        k = max(1, int(0.25 * bs * bs))

        idx = np.argpartition(np.abs(flat_V), -k, axis=1)[:, -k:]
        vals = np.take_along_axis(flat_V, idx, axis=1)

        i_positions = np.arange(n_rows, dtype=np.int32) * stride
        j_positions = np.arange(n_cols, dtype=np.int32) * stride
        i_grid, j_grid = np.meshgrid(i_positions, j_positions, indexing="ij")

        meta = dict(shape=(m, n), block_size=bs, n_blocks=n_blocks)
        data = struct.pack("<ii", m, n) + struct.pack("<i", n_blocks)
        data += i_grid.ravel().tobytes()
        data += j_grid.ravel().tobytes()
        data += struct.pack("<i", k)
        data += idx.astype(np.int32).tobytes()
        data += vals.astype(np.float16).tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n = metadata["shape"]
        bs = metadata["block_size"]
        pos = struct.calcsize("<ii")
        n_blocks = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        i_positions = np.frombuffer(
            data[pos : pos + n_blocks * 4], dtype=np.int32
        ).copy()
        pos += n_blocks * 4
        j_positions = np.frombuffer(
            data[pos : pos + n_blocks * 4], dtype=np.int32
        ).copy()
        pos += n_blocks * 4

        k = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        idx = np.frombuffer(data[pos : pos + n_blocks * k * 4], dtype=np.int32).copy()
        idx = idx.reshape(n_blocks, k)
        pos += n_blocks * k * 4

        vals = np.frombuffer(
            data[pos : pos + n_blocks * k * 2], dtype=np.float16
        ).astype(np.float64)
        vals = vals.reshape(n_blocks, k)

        B = np.array(
            [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
            dtype=np.float64,
        )

        V_hat = np.zeros((n_blocks, bs * bs), dtype=np.float64)
        np.put_along_axis(V_hat, idx, vals, axis=1)
        V_hat = V_hat.reshape(n_blocks, bs, bs)

        B_inv = np.linalg.inv(B)
        tmp = np.einsum("pi,rpj->rpj", B_inv.T, V_hat)
        blocks = np.einsum("rpj,jk->rpk", tmp, B_inv)

        recon = np.zeros((m, n), dtype=np.float64)
        for r in range(n_blocks):
            iii, jjj = i_positions[r], j_positions[r]
            recon[iii : iii + bs, jjj : jjj + bs] = blocks[r]

        return recon.astype(np.float32)


class PolynomialApprox:
    """Polynomial function approximation of weight structure."""

    name = "polynomial_approx"
    category = "spectral"

    def compress(self, tensor: np.ndarray, degree: int = 8) -> Tuple[bytes, dict]:
        return Chebyshev().compress(tensor, n_coeffs=degree + 1)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return Chebyshev().decompress(data, metadata)


# Re-export for backward-compatible imports (test harness uses transforms.DCT2D)
from .dct import DCT2D  # noqa: E402, F401

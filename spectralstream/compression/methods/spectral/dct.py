"""HPC DCT — FFT-based (O(N log N)), vectorized block processing via reshape + _dct_via_fft_1d.

Novel methods:
  - DCTQuant:         2D DCT decorrelation + block-int4 quantization of ALL coefficients
  - SpectralDCTHybrid:DCT → sort by magnitude → delta-encoded sparse indices → quantized values
  - DCTSparseQuant:   Block DCT → per-block top-k → sparse bitmask indices → quantized values
"""

from __future__ import annotations

import gc
import math
import struct
from typing import Any, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    auto_keep_fraction,
    dct,
    dct_2d,
    idct,
    idct_2d,
)
from spectralstream.core.math_primitives.transforms import (
    _dct_via_fft_1d,
    _idct_via_matrix_1d,
)


def _delta_encode(arr: np.ndarray) -> np.ndarray:
    """Delta-encode a sorted integer array for efficient storage."""
    if arr.size == 0:
        return arr
    delta = np.empty(arr.size, dtype=np.int32)
    delta[0] = arr[0]
    np.subtract(arr[1:], arr[:-1], out=delta[1:])
    return delta


def _delta_decode(delta: np.ndarray) -> np.ndarray:
    """Decode delta-encoded array back to absolute indices."""
    return np.cumsum(delta, dtype=np.int64).astype(np.intp)


def _zigzag_sort_key(shape: Tuple[int, int]) -> np.ndarray:
    """Return linear indices sorted by zigzag order for a given shape."""
    m, n = shape
    idx = np.arange(m * n, dtype=np.int32).reshape(m, n)
    i_indices, j_indices = np.indices((m, n))
    s = i_indices + j_indices
    even_mask = s % 2 == 0
    primary = np.where(even_mask, j_indices, i_indices) + (m + n) * s.astype(np.float64)
    secondary = np.where(even_mask, i_indices, j_indices)
    order = np.lexsort((secondary.ravel(), primary.ravel()))
    return idx.ravel()[order]


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


class DCTQuant:
    """DCT + block-int4 quantization of ALL coefficients (spectral decorrelation + quantization).

    O(N log N) via FFT-based DCT, then block-int4 quantization on decorrelated coefficients.
    """

    name = "dct_quant"
    category = "spectral"

    def __init__(self, quant_bits: int = 4):
        self.quant_bits = quant_bits

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 64,
        bits: int | None = None,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        ndim = orig.ndim
        if ndim == 1:
            orig = orig.reshape(1, -1)
        bits_ = bits if bits is not None else self.quant_bits
        qmax = float(2 ** (bits_ - 1) - 1)

        coeffs = dct_2d(orig)
        flat = coeffs.ravel()
        n = len(flat)
        bs = block_size
        padded_n = int(math.ceil(n / bs) * bs)
        padded = np.zeros(padded_n, dtype=np.float64)
        padded[:n] = flat
        blocks = padded.reshape(-1, bs)
        n_blocks = blocks.shape[0]

        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-10, amax / qmax, 1.0)
        quantized = np.clip(np.round(blocks / scales[:, None]), -qmax - 1, qmax).astype(
            np.int8 if bits_ <= 8 else np.int16
        )

        meta = dict(
            shape=orig.shape,
            block_size=bs,
            quant_bits=bits_,
            n_blocks=n_blocks,
            ndim=ndim,
        )
        data = (
            struct.pack("<ii", *orig.shape)
            + struct.pack("<ii", n_blocks, bs)
            + struct.pack("<i", bits_)
            + scales.astype(np.float16).tobytes()
            + quantized.tobytes()
        )
        del coeffs, flat, blocks, quantized
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        ndim = metadata.get("ndim", 2)
        pos = struct.calcsize("<ii")
        n_blocks = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        bs = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        bits_ = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        dtype_quant = np.int8 if bits_ <= 8 else np.int16

        scales = np.frombuffer(data[pos : pos + n_blocks * 2], dtype=np.float16).astype(
            np.float64
        )
        pos += n_blocks * 2
        qsize = n_blocks * bs
        quantized = np.frombuffer(
            data[pos : pos + qsize * np.dtype(dtype_quant).itemsize], dtype=dtype_quant
        ).astype(np.float64)
        quantized = quantized.reshape(n_blocks, bs)

        flat = (quantized * scales[:, None]).ravel()
        flat = flat[: shape[0] * shape[1]]
        coeffs = flat.reshape(shape)
        recon = idct_2d(coeffs).astype(np.float32)
        if ndim == 1:
            recon = recon.ravel()
        return recon


class DCTAdaptiveBits:
    """JPEG-style spectral compression: DCT + per-frequency bit allocation."""

    name = "dct_adaptive_bits"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 16,
        target_ratio: float = 8.0,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        ndim = orig.ndim
        if ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape
        bs = block_size

        n_rows = (m + bs - 1) // bs
        n_cols = (n + bs - 1) // bs
        pm = n_rows * bs
        pn = n_cols * bs
        padded = np.zeros((pm, pn), dtype=np.float64)
        padded[:m, :n] = orig

        blocks = (
            padded.reshape(n_rows, bs, n_cols, bs)
            .transpose(0, 2, 1, 3)
            .reshape(-1, bs, bs)
        )
        n_blocks = blocks.shape[0]

        dct_rows = _dct_via_fft_1d(blocks)
        dct_cols = _dct_via_fft_1d(dct_rows.transpose(0, 2, 1)).transpose(0, 2, 1)

        zigzag = _zigzag_sort_key((bs, bs))
        flat_zz = dct_cols.reshape(n_blocks, bs * bs)[:, zigzag]

        n_coeff = bs * bs
        zigzag_frac = np.arange(n_coeff, dtype=np.float64) / n_coeff
        bits_per_coeff = np.maximum(2, (8 - zigzag_frac * 6)).astype(np.int32)

        scales_2d = np.max(np.abs(dct_cols.reshape(n_blocks, -1)), axis=0)
        scales_2d = np.where(scales_2d < 1e-10, 1.0, scales_2d)
        scales_2d_zz = scales_2d[zigzag]

        data = struct.pack("<ii", m, n)
        data += struct.pack("<ii", n_blocks, bs)
        data += struct.pack("<d", target_ratio)
        data += scales_2d.astype(np.float16).tobytes()
        data += bits_per_coeff.astype(np.int8).tobytes()

        for c in range(n_coeff):
            bpc = int(bits_per_coeff[c])
            if bpc < 1:
                continue
            qmax = float(2 ** (bpc - 1) - 1)
            col = flat_zz[:, c].copy()
            sc = scales_2d_zz[c]
            if sc < 1e-10:
                data += np.zeros(n_blocks, dtype=np.int8).tobytes()
            else:
                qcol = np.clip(np.round(col / sc * qmax), -qmax - 1, qmax).astype(
                    np.int8
                )
                data += qcol.tobytes()

        meta = dict(
            shape=orig.shape,
            block_size=bs,
            n_blocks=n_blocks,
            n_coeff=n_coeff,
            target_ratio=target_ratio,
            ndim=ndim,
            bits_per_coeff=bits_per_coeff.tolist(),
        )
        del blocks, dct_rows, dct_cols, flat_zz
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bs = metadata["block_size"]
        n_blocks = metadata["n_blocks"]
        n_coeff = metadata["n_coeff"]
        bits_per_coeff = np.array(metadata["bits_per_coeff"], dtype=np.int32)
        ndim = metadata.get("ndim", 2)

        pos = 8
        _n_blocks = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        _bs = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        _target = struct.unpack_from("<d", data, pos)[0]
        pos += 8

        scales_2d = np.frombuffer(
            data[pos : pos + n_coeff * 2], dtype=np.float16
        ).astype(np.float64)
        pos += n_coeff * 2
        _bpc_check = np.frombuffer(data[pos : pos + n_coeff], dtype=np.int8).copy()
        pos += n_coeff

        zigzag_inv = np.argsort(_zigzag_sort_key((bs, bs)))
        flat_zz = np.zeros((n_blocks, n_coeff), dtype=np.float64)
        for c in range(n_coeff):
            bpc = int(bits_per_coeff[c])
            if bpc < 1:
                continue
            qmax = float(2 ** (bpc - 1) - 1)
            col_raw = np.frombuffer(data[pos : pos + n_blocks], dtype=np.int8).astype(
                np.float64
            )
            pos += n_blocks
            sc = scales_2d[c]
            flat_zz[:, c] = col_raw / qmax * sc

        flat_zig = flat_zz[:, zigzag_inv]
        V_hat = flat_zig.reshape(n_blocks, bs, bs)

        idct_cols = _idct_via_matrix_1d(V_hat.transpose(0, 2, 1)).transpose(0, 2, 1)
        idct_rows = _idct_via_matrix_1d(idct_cols)

        recon = np.zeros(shape, dtype=np.float64)
        n_cols_grid = (shape[1] + bs - 1) // bs
        for r in range(n_blocks):
            ri = r // n_cols_grid
            rj = r % n_cols_grid
            iii = ri * bs
            jjj = rj * bs
            if iii >= shape[0] or jjj >= shape[1]:
                continue
            bh = min(bs, shape[0] - iii)
            bw = min(bs, shape[1] - jjj)
            recon[iii : iii + bh, jjj : jjj + bw] = idct_rows[r, :bh, :bw]

        out = recon.astype(np.float32)
        if ndim == 1:
            out = out.ravel()
        return out


class SpectralDCTHybrid:
    """Spectral-structural hybrid: DCT → global sort by magnitude → delta-encoded indices + quantized values.

    Full algorithm:
      1. 2D DCT decorrelation
      2. Global magnitude sort of coefficients
      3. Top-k retention from sorted list
      4. Re-sort kept coefficients by original index
      5. Delta-encode the re-sorted indices
      6. Block-int4 quantize the values
    """

    name = "spectral_dct_hybrid"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_fraction: float = 0.05,
        block_size: int = 32,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        ndim = orig.ndim
        if ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape

        coeffs = dct_2d(orig)
        flat = coeffs.ravel()
        n_total = flat.size
        k = max(1, int(keep_fraction * n_total))

        mag = np.abs(flat)
        top_idx = np.argpartition(mag, -k)[-k:]
        top_vals = flat[top_idx].copy()

        sort_order = np.argsort(top_idx)
        top_idx = top_idx[sort_order]
        top_vals = top_vals[sort_order]
        del sort_order

        deltas = _delta_encode(top_idx.astype(np.int32))
        qmax = 7.0
        top_vals_abs = np.abs(top_vals)
        signed = np.sign(top_vals)
        amax = np.max(top_vals_abs)
        scale = amax / qmax if amax > 1e-10 else 1.0
        quantized = np.clip(np.round(top_vals_abs / scale), 0, 7).astype(np.int8)
        quantized = quantized * signed.astype(np.int8)

        meta = dict(
            shape=orig.shape,
            keep_fraction=keep_fraction,
            n_kept=k,
            n_total=n_total,
            ndim=ndim,
            scale=float(scale),
        )
        data = (
            struct.pack("<ii", m, n)
            + struct.pack("<i", k)
            + struct.pack("<d", scale)
            + deltas.tobytes()
            + quantized.tobytes()
        )
        del coeffs, flat, top_idx, top_vals, deltas
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        ndim = metadata.get("ndim", 2)
        k = metadata["n_kept"]
        scale = metadata["scale"]
        pos = struct.calcsize("<ii")
        k_check = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        scale_check = struct.unpack_from("<d", data, pos)[0]
        pos += 8

        deltas = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy()
        pos += k * 4
        quantized = np.frombuffer(data[pos : pos + k], dtype=np.int8).astype(np.float64)
        signed = np.sign(quantized)
        vals = np.abs(quantized) * scale * signed

        indices = _delta_decode(deltas)
        coeffs = np.zeros(shape[0] * shape[1], dtype=np.float64)
        valid_mask = indices < coeffs.size
        coeffs[indices[valid_mask]] = vals[valid_mask]

        recon = idct_2d(coeffs.reshape(shape)).astype(np.float32)
        if ndim == 1:
            recon = recon.ravel()
        return recon


class DCTSparseQuant:
    """Block DCT + per-block top-k sparse indices (bitmask) + int4 quantized values.

    Each block DCT coefficient is either kept (quantized to int4) or zeroed.
    Indices stored as dense bitmask for efficient decode.
    """

    name = "dct_sparse_quant"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 16,
        keep_density: float = 0.125,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        ndim = orig.ndim
        if ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape
        bs = block_size

        n_rows = (m + bs - 1) // bs
        n_cols = (n + bs - 1) // bs
        pm = n_rows * bs
        pn = n_cols * bs
        padded = np.zeros((pm, pn), dtype=np.float64)
        padded[:m, :n] = orig

        blocks = (
            padded.reshape(n_rows, bs, n_cols, bs)
            .transpose(0, 2, 1, 3)
            .reshape(-1, bs, bs)
        )
        n_blocks = blocks.shape[0]

        dct_rows = _dct_via_fft_1d(blocks)
        dct_cols = _dct_via_fft_1d(dct_rows.transpose(0, 2, 1)).transpose(0, 2, 1)

        flat_all = dct_cols.reshape(n_blocks, bs * bs)
        k_per_block = max(1, int(keep_density * bs * bs))
        idx = np.argpartition(np.abs(flat_all), -k_per_block, axis=1)[:, -k_per_block:]
        vals = np.take_along_axis(flat_all, idx, axis=1)

        bitmask = np.zeros((n_blocks, bs * bs), dtype=np.uint8)
        np.put_along_axis(bitmask, idx, 1, axis=1)
        bitmask_packed = np.packbits(bitmask, axis=1)

        qmax = 7.0
        vals_abs = np.abs(vals)
        amax = np.max(vals_abs, axis=1, keepdims=True)
        scales = np.where(amax > 1e-10, amax / qmax, 1.0).astype(np.float32)
        quant = np.clip(np.round(vals_abs / scales), 0, 7).astype(np.int8)
        quant = quant * np.sign(vals).astype(np.int8)

        i_pos = (np.arange(n_rows) * bs).astype(np.int32)
        j_pos = (np.arange(n_cols) * bs).astype(np.int32)
        i_grid, j_grid = np.meshgrid(i_pos, j_pos, indexing="ij")

        meta = dict(
            shape=orig.shape,
            block_size=bs,
            n_blocks=n_blocks,
            keep_density=keep_density,
            ndim=ndim,
        )
        data = (
            struct.pack("<ii", m, n)
            + struct.pack("<i", n_blocks)
            + i_grid.ravel().tobytes()
            + j_grid.ravel().tobytes()
            + bitmask_packed.tobytes()
            + scales.tobytes()
            + quant.tobytes()
        )
        del blocks, dct_rows, dct_cols, flat_all, bitmask, bitmask_packed
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bs = metadata["block_size"]
        n_blocks = metadata["n_blocks"]
        pos = struct.calcsize("<ii")
        n_blocks_check = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        i_flat = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.int32).copy()
        pos += n_blocks * 4
        j_flat = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.int32).copy()
        pos += n_blocks * 4

        packed_bytes = (bs * bs + 7) // 8
        bitmask_packed = (
            np.frombuffer(data[pos : pos + n_blocks * packed_bytes], dtype=np.uint8)
            .copy()
            .reshape(n_blocks, packed_bytes)
        )
        pos += n_blocks * packed_bytes
        bitmask = np.unpackbits(bitmask_packed, axis=1)[:, : bs * bs]

        scales = (
            np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
            .astype(np.float64)
            .reshape(-1, 1)
        )
        pos += n_blocks * 4

        k_per_block = int(np.sum(bitmask[0]))
        quant = (
            np.frombuffer(data[pos : pos + n_blocks * k_per_block], dtype=np.int8)
            .astype(np.float64)
            .reshape(n_blocks, k_per_block)
        )

        signed = np.sign(quant)
        vals = np.abs(quant) * scales * signed
        V_hat = np.zeros((n_blocks, bs * bs), dtype=np.float64)
        val_offset = 0
        for b in range(n_blocks):
            mask_b = bitmask[b].astype(bool)
            nk = int(np.sum(mask_b))
            V_hat[b, mask_b] = vals[b, :nk]

        V_hat = V_hat.reshape(n_blocks, bs, bs)
        idct_cols = _idct_via_matrix_1d(V_hat.transpose(0, 2, 1)).transpose(0, 2, 1)
        idct_rows = _idct_via_matrix_1d(idct_cols)

        recon = np.zeros(shape, dtype=np.float64)
        for r in range(n_blocks):
            iii, jjj = i_flat[r], j_flat[r]
            recon[iii : iii + bs, jjj : jjj + bs] = idct_rows[
                r, : min(bs, shape[0] - iii), : min(bs, shape[1] - jjj)
            ]

        out = recon.astype(np.float32)
        if metadata.get("ndim", 2) == 1:
            out = out.ravel()
        return out


class DCTBlock:
    """DCT block compression — vectorized block processing via reshape."""

    name = "dct_block"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 32,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
        ndim = orig.ndim
        if ndim == 1:
            orig = orig.reshape(1, -1)
        m, n = orig.shape
        bs = block_size

        # Pad to multiples of bs for clean block decomposition
        n_rows = (m + bs - 1) // bs
        n_cols = (n + bs - 1) // bs
        pm = n_rows * bs
        pn = n_cols * bs
        padded = np.zeros((pm, pn), dtype=np.float64)
        padded[:m, :n] = orig

        # Extract all non-overlapping blocks as a batch
        # Shape: (n_rows, bs, n_cols, bs) -> (n_rows * n_cols, bs, bs)
        blocks = (
            padded.reshape(n_rows, bs, n_cols, bs)
            .transpose(0, 2, 1, 3)
            .reshape(-1, bs, bs)
        )
        n_blocks = blocks.shape[0]

        # Batched 2D DCT via _dct_via_fft_1d (operates on last axis of ND array)
        # Step 1: DCT along rows of each block (axis=2 of blocks = last axis)
        dct_rows = _dct_via_fft_1d(blocks)  # (n_blocks, bs, bs)
        # Step 2: DCT along columns of each block (axis=1)
        dct_cols = _dct_via_fft_1d(dct_rows.transpose(0, 2, 1)).transpose(0, 2, 1)

        # Flatten all DCT blocks: (n_blocks, bs*bs)
        flat_all = dct_cols.reshape(n_blocks, bs * bs)

        # Compute keep_fraction from all coefficients
        all_concat = flat_all.ravel()
        if keep_fraction is None:
            kf = auto_keep_fraction(all_concat, target_energy)
        else:
            kf = keep_fraction
        del all_concat

        # Top-k per block (vectorized)
        k_per_block = max(1, int(kf * bs * bs))
        idx = np.argpartition(np.abs(flat_all), -k_per_block, axis=1)[:, -k_per_block:]
        vals = np.take_along_axis(flat_all, idx, axis=1)

        # Build block position arrays
        i_pos = (np.arange(n_rows) * bs).astype(np.int32)
        j_pos = (np.arange(n_cols) * bs).astype(np.int32)
        i_grid, j_grid = np.meshgrid(i_pos, j_pos, indexing="ij")
        i_flat = i_grid.ravel()
        j_flat = j_grid.ravel()
        # bh, bw per block (edge blocks may be smaller)
        bh_arr = np.minimum(bs * np.ones(n_blocks, dtype=np.int32), m - i_flat)
        bw_arr = np.minimum(bs * np.ones(n_blocks, dtype=np.int32), n - j_flat)

        meta = dict(
            shape=orig.shape,
            block_size=bs,
            keep_fraction=kf,
            target_energy=target_energy,
            n_blocks=n_blocks,
            ndim=ndim,
        )

        # Serialize: header + block positions + indices + values
        data = struct.pack("<ii", m, n) + struct.pack("<i", n_blocks)
        data += i_flat.tobytes()
        data += j_flat.tobytes()
        data += bh_arr.tobytes()
        data += bw_arr.tobytes()
        data += struct.pack("<i", k_per_block)
        data += idx.astype(np.int32).tobytes()
        data += vals.astype(np.float16).tobytes()

        del blocks, dct_rows, dct_cols, flat_all, idx, vals
        gc.collect()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        m, n = shape
        bs = metadata["block_size"]
        pos = struct.calcsize("<ii")
        n_blocks = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        i_flat = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.int32).copy()
        pos += n_blocks * 4
        j_flat = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.int32).copy()
        pos += n_blocks * 4
        bh_arr = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.int32).copy()
        pos += n_blocks * 4
        bw_arr = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.int32).copy()
        pos += n_blocks * 4

        k_per_block = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        idx = np.frombuffer(
            data[pos : pos + n_blocks * k_per_block * 4], dtype=np.int32
        ).copy()
        idx = idx.reshape(n_blocks, k_per_block)
        pos += n_blocks * k_per_block * 4

        vals = np.frombuffer(
            data[pos : pos + n_blocks * k_per_block * 2], dtype=np.float16
        ).astype(np.float64)
        vals = vals.reshape(n_blocks, k_per_block)

        # Reconstruct all blocks at once
        V_hat = np.zeros((n_blocks, bs * bs), dtype=np.float64)
        np.put_along_axis(V_hat, idx, vals, axis=1)
        V_hat = V_hat.reshape(n_blocks, bs, bs)

        # Batched 2D IDCT via _idct_via_matrix_1d (operates on last axis of ND)
        # Step 1: IDCT along columns (transpose so columns become last axis)
        idct_cols = _idct_via_matrix_1d(V_hat.transpose(0, 2, 1)).transpose(0, 2, 1)
        # Step 2: IDCT along rows
        idct_rows = _idct_via_matrix_1d(idct_cols)

        # Scatter blocks into output
        recon = np.zeros((m, n), dtype=np.float64)
        for r in range(n_blocks):
            iii, jjj = i_flat[r], j_flat[r]
            recon[iii : iii + bh_arr[r], jjj : jjj + bw_arr[r]] = idct_rows[
                r, : bh_arr[r], : bw_arr[r]
            ]

        out = recon.astype(np.float32)
        if metadata.get("ndim", 2) == 1:
            out = out.ravel()
        del V_hat, idct_cols, idct_rows
        gc.collect()
        return out


class DCT2D:
    """2D DCT with global coefficient thresholding — uses FFT-based DCT."""

    name = "dct_2d"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        orig = tensor.astype(np.float64)
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
        shape = metadata["shape"]
        pos = struct.calcsize("<ii")
        k = metadata["n_kept"]
        idx = np.frombuffer(data[pos : pos + k * 4], dtype=np.int32).copy().astype(int)
        pos += k * 4
        vals = np.frombuffer(data[pos : pos + k * 2], dtype=np.float16).astype(
            np.float64
        )
        coeffs = np.zeros(shape[0] * shape[1], dtype=np.float64)
        coeffs[idx] = vals
        return idct_2d(coeffs.reshape(shape)).astype(np.float32)


class DCT2DBlock:
    """2D DCT block compression — delegates to DCTBlock with FFT-based DCT."""

    name = "dct_2d_block"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        block_size: int = 8,
        keep_fraction: float | None = None,
        target_energy: float = 0.99,
    ) -> Tuple[bytes, dict]:
        return DCTBlock().compress(
            tensor,
            block_size=block_size,
            keep_fraction=keep_fraction,
            target_energy=target_energy,
        )

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return DCTBlock().decompress(data, metadata)

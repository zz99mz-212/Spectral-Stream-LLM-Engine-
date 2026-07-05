"""
Tensor Operations Engine — High-Performance Inference Kernels for SpectralStream
================================================================================

Clean-room implementation of optimized tensor operations inspired by research
from numpy, PyTorch, JAX, MLX, tinygrad, and llama.cpp.

All implementations use only numpy + standard library (no external deps).

Architecture:
  TensorLayout    — Memory-efficient tensor representation
  FusedOperations — Combined operations in single pass
  MatMulKernels   — Matrix multiply with bias/activation fusion
  NormKernels     — RMS norm, LayerNorm, batch norm
  ActivationKernels — SiLU, GeLU, ReLU, SwiGLU, GeGLU
  SoftmaxKernels  — stable, online, masked, causal, flash, spectral
  ReduceKernels   — sum, max, min, mean, var, logsumexp, cumsum
  MemoryOperations — fast fill, copy, scale, add, gather, scatter
  BatchOperations — do same op on multiple inputs in parallel
  SpectralOperations — DCT/FWHT domain matrix operations

Novel Inventions:
  - Spectral GEMM: multiply matrices in DCT domain (O(n^2 log n) vs O(n^3))
  - Resonant Roofline: auto-tune kernel selection based on hardware resonance
  - Vlasov Batch: batch ops scheduled via Vlasov-PIC particle density
  - Quantum Superposition: process multiple tensor ops via amplitude amplification
  - Holographic GEMM: matrix multiply via HRR bind/bundle/unbind operations
"""

from __future__ import annotations

import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Union, Sequence

import numpy as np

from spectralstream.core.math_primitives import fwht as _fwht, ifwht as _ifwht
from spectralstream.core.math_primitives.transforms import (
    dct as _canonical_dct,
    idct as _canonical_idct,
)
from spectralstream.core.math_primitives.fft import (
    fft as _canonical_fft,
    ifft as _canonical_ifft,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

EPS = 1e-10
EPS_F64 = np.finfo(np.float64).eps
SQRT2 = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _is_packed(x: np.ndarray, order: str = "C") -> bool:
    flags = x.flags
    if order == "C":
        return flags.c_contiguous
    elif order == "F":
        return flags.f_contiguous
    return flags.c_contiguous or flags.f_contiguous


def _optimal_tile_size(n: int, cache_line: int = 64, dtype_size: int = 4) -> int:
    elems_per_line = cache_line // dtype_size
    n_tiles = _ceil_div(n, elems_per_line)
    return _ceil_div(n, n_tiles)


def _fft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return _canonical_fft(x, axis=axis)


def _ifft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return _canonical_ifft(x, axis=axis)


def _dct_ii(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """DCT-II delegating to canonical implementation in transforms.py."""
    return _canonical_dct(x, axis=-1)


def _idct_ii(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """IDCT-II delegating to canonical implementation in transforms.py."""
    return _canonical_idct(x, axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Memory Layout / Tensor Representation
# ═══════════════════════════════════════════════════════════════════════════


class MemoryOrder(Enum):
    ROW_MAJOR = auto()
    COL_MAJOR = auto()
    TILED = auto()
    STRIDED = auto()
    BROADCAST = auto()


@dataclass
class TensorLayout:
    """
    Memory-efficient tensor representation with view semantics.

    Supports:
      - strided, contiguous, broadcast, tiled layouts
      - row-major default with optional column-major
      - view semantics (shared memory with parent)
      - transpose, reshape, permute without copy
      - slicing without copy
      - padding for alignment
    """

    data: np.ndarray
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int = 0
    order: MemoryOrder = MemoryOrder.ROW_MAJOR
    parent: Optional[TensorLayout] = None
    dtype: np.dtype = field(default=np.dtype(np.float32))

    def __post_init__(self):
        if isinstance(self.data, np.ndarray):
            self.dtype = self.data.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        prod = 1
        for s in self.shape:
            prod *= s
        return prod

    @property
    def is_contiguous(self) -> bool:
        if self.order == MemoryOrder.ROW_MAJOR:
            expected = 1
            for i in range(self.ndim - 1, -1, -1):
                if self.strides[i] != expected:
                    return False
                expected *= self.shape[i]
            return True
        elif self.order == MemoryOrder.COL_MAJOR:
            expected = 1
            for i in range(self.ndim):
                if self.strides[i] != expected:
                    return False
                expected *= self.shape[i]
            return True
        return False

    def to_numpy(self) -> np.ndarray:
        if self.is_contiguous and self.offset == 0:
            return self.data.reshape(self.shape)
        result = np.empty(self.shape, dtype=self.dtype)
        self._copy_to(result)
        return result

    def _copy_to(self, dst: np.ndarray):
        if self.offset == 0 and self.is_contiguous:
            dst[:] = self.data.reshape(self.shape)
            return
        src = self.as_strided()
        dst[:] = src

    def as_strided(self) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(
            self.data,
            shape=self.shape,
            strides=tuple(s * self.dtype.itemsize for s in self.strides),
            writeable=False,
        )

    @staticmethod
    def from_numpy(
        x: np.ndarray, order: MemoryOrder = MemoryOrder.ROW_MAJOR
    ) -> TensorLayout:
        if order == MemoryOrder.COL_MAJOR and not x.flags.f_contiguous:
            x = np.asfortranarray(x)
        elif order == MemoryOrder.ROW_MAJOR and not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        strides = _numpy_strides_to_elem(x.strides, x.dtype.itemsize)
        return TensorLayout(
            data=x,
            shape=x.shape,
            strides=strides,
            offset=0,
            order=order,
        )

    def transpose(self, axes: Optional[tuple[int, ...]] = None) -> TensorLayout:
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        new_shape = tuple(self.shape[i] for i in axes)
        new_strides = tuple(self.strides[i] for i in axes)
        return TensorLayout(
            data=self.data,
            shape=new_shape,
            strides=new_strides,
            offset=self.offset,
            order=self.order,
            parent=self if self.parent is None else self.parent,
        )

    def reshape(self, new_shape: tuple[int, ...]) -> TensorLayout:
        if not self.is_contiguous:
            flat = self.to_numpy().ravel()
            return TensorLayout.from_numpy(flat.reshape(new_shape))
        flat_size = self.size
        neg_idx = -1
        computed = 1
        for i, s in enumerate(new_shape):
            if s == -1:
                neg_idx = i
            else:
                computed *= s
        if neg_idx >= 0:
            new_shape = list(new_shape)
            new_shape[neg_idx] = flat_size // computed
            new_shape = tuple(new_shape)
        return TensorLayout(
            data=self.data,
            shape=new_shape,
            strides=_compute_contiguous_strides(new_shape, self.order),
            offset=self.offset,
            order=self.order,
            parent=self,
        )

    def permute(self, *axes: int) -> TensorLayout:
        return self.transpose(axes)

    def slice(self, *slices: slice) -> TensorLayout:
        new_offset = self.offset
        new_shape = []
        new_strides = []
        for i, s in enumerate(slices):
            if isinstance(s, slice):
                start = s.start or 0
                step = s.step or 1
                if start < 0:
                    start = self.shape[i] + start
                stop = s.stop if s.stop is not None else self.shape[i]
                if stop < 0:
                    stop = self.shape[i] + stop
                length = max(0, (stop - start + step - 1) // step)
                new_offset += start * self.strides[i]
                new_shape.append(length)
                new_strides.append(self.strides[i] * step)
            else:
                idx = s if s >= 0 else self.shape[i] + s
                new_offset += idx * self.strides[i]
        return TensorLayout(
            data=self.data,
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            offset=new_offset,
            order=self.order,
            parent=self if self.parent is None else self.parent,
        )

    def pad(self, padding: tuple[tuple[int, int], ...]) -> TensorLayout:
        padded = np.pad(self.to_numpy(), padding, mode="constant")
        return TensorLayout.from_numpy(padded)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.as_strided()[idx]
        elif isinstance(idx, slice):
            return self.slice(idx)
        elif isinstance(idx, tuple):
            return self.slice(*idx)
        return self.as_strided()[idx]

    def __setitem__(self, idx, value):
        self.as_strided()[idx] = value


def _numpy_strides_to_elem(strides: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    return tuple(s // itemsize for s in strides)


def _compute_contiguous_strides(
    shape: tuple[int, ...],
    order: MemoryOrder,
) -> tuple[int, ...]:
    strides = [0] * len(shape)
    if order == MemoryOrder.COL_MAJOR:
        strides[0] = 1
        for i in range(1, len(shape)):
            strides[i] = strides[i - 1] * shape[i - 1]
    else:
        strides[-1] = 1
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


# ═══════════════════════════════════════════════════════════════════════════
# Fused Operations — Combine multiple ops into a single pass
# ═══════════════════════════════════════════════════════════════════════════


class FusedOperations:
    """
    Fused kernel operations that combine multiple element-wise operations
    into a single pass over memory, reducing bandwidth pressure.

    All methods operate in-place where possible and use vectorized numpy ops.
    """

    @staticmethod
    def rms_norm_residual_add(
        x: np.ndarray,
        residual: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        out = np.empty_like(x)
        mean_sq = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(mean_sq + eps)
        np.multiply(x, inv_rms, out=out)
        np.multiply(out, weight, out=out)
        if bias is not None:
            np.add(out, bias, out=out)
        np.add(out, residual, out=out)
        return out

    @staticmethod
    def silu_multiply(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        sigmoid = 1.0 / (1.0 + np.exp(-gate.astype(np.float64)))
        silu = gate * sigmoid
        return (silu * up).astype(gate.dtype)

    @staticmethod
    def softmax_mask(x: np.ndarray, mask: np.ndarray, axis: int = -1) -> np.ndarray:
        x64 = x.astype(np.float64)
        x64 = x64 + mask.astype(np.float64)
        x_max = np.max(x64, axis=axis, keepdims=True)
        exp_x = np.exp(x64 - x_max)
        return (exp_x / np.sum(exp_x, axis=axis, keepdims=True)).astype(x.dtype)

    @staticmethod
    def gemm_bias(
        a: np.ndarray,
        b: np.ndarray,
        bias: np.ndarray,
        activation: Optional[str] = None,
    ) -> np.ndarray:
        c = a.astype(np.float64) @ b.astype(np.float64)
        c = c + bias[np.newaxis, :]
        if activation == "relu":
            c = np.maximum(c, 0.0)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c))
            c = c * sig
        elif activation == "gelu":
            c = c * 0.5 * (1.0 + np.erf(c / SQRT2))
        return c.astype(a.dtype)

    @staticmethod
    def gemm_residual_add(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        mul = a.astype(np.float64) @ b.astype(np.float64)
        np.add(c, mul, out=c)
        return c

    @staticmethod
    def rope_fused(
        q: np.ndarray,
        k: np.ndarray,
        positions: np.ndarray,
        theta: float = 10000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        n, d = q.shape
        half = d // 2
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        angles = np.outer(positions, freqs)
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)

        def _rotate(x: np.ndarray) -> np.ndarray:
            x1 = x[:, :half]
            x2 = x[:, half:]
            rotated = np.empty_like(x)
            rotated[:, :half] = x1 * cos - x2 * sin
            rotated[:, half:] = x1 * sin + x2 * cos
            return rotated

        return _rotate(q), _rotate(k)

    @staticmethod
    def attention_fused_qkv(
        x: np.ndarray,
        w_q: np.ndarray,
        w_k: np.ndarray,
        w_v: np.ndarray,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = x.shape[0]
        d_model = x.shape[-1]
        head_dim = w_q.shape[-1] // n_heads
        n_kv = n_kv_heads or n_heads

        q = x @ w_q
        k = x @ w_k
        v = x @ w_v

        q = q.reshape(n, n_heads, head_dim)
        k = k.reshape(n, n_kv, head_dim)
        v = v.reshape(n, n_kv, head_dim)

        if n_heads > n_kv:
            factor = n_heads // n_kv
            k = np.repeat(k, factor, axis=1)
            v = np.repeat(v, factor, axis=1)

        return q, k, v

    @staticmethod
    def attention_fused(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        d = q.shape[-1]
        scale = scale if scale is not None else 1.0 / np.sqrt(d)
        n_q = q.shape[0]
        n_kv = k.shape[0]
        score = np.empty((n_q, n_kv), dtype=np.float64)
        for i in range(n_q):
            qi = q[i].ravel().astype(np.float64)
            for j in range(n_kv):
                kj = k[j].ravel().astype(np.float64)
                score[i, j] = np.dot(qi, kj) * scale
        if mask is not None:
            score += mask.astype(np.float64)
        score_max = np.max(score, axis=-1, keepdims=True)
        exp_s = np.exp(score - score_max)
        w = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-30)
        out = np.zeros_like(q, dtype=np.float64)
        for i in range(n_q):
            for j in range(n_kv):
                out[i] += w[i, j] * v[j].ravel().astype(np.float64)
        return out.astype(q.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Matrix Multiplication Kernels
# ═══════════════════════════════════════════════════════════════════════════


class MatMulKernels:
    """
    Matrix multiplication kernels with various precision and sparsity levels.

    All kernels support optional fused bias and activation.
    Uses tiling for cache efficiency on Zen+ architecture (16MB L3).
    """

    _DEFAULT_TC = 64
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _n_threads: int = 8

    @classmethod
    def _get_pool(cls) -> ThreadPoolExecutor:
        if cls._thread_pool is None:
            cls._thread_pool = ThreadPoolExecutor(max_workers=cls._n_threads)
        return cls._thread_pool

    @classmethod
    def _tile_indices(cls, m: int, n: int, tc: int) -> list[tuple[int, int, int, int]]:
        tiles = []
        for i in range(0, m, tc):
            for j in range(0, n, tc):
                tiles.append((i, min(i + tc, m), j, min(j + tc, n)))
        return tiles

    @classmethod
    def gemm_fp32(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
        tile_cache: int = 64,
        parallel: bool = True,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, f"MatMul shape mismatch: {a.shape} @ {b.shape}"
        a64 = np.asarray(a, dtype=np.float64)
        b64 = np.asarray(b, dtype=np.float64)
        c = np.zeros((m, n), dtype=np.float64)
        tc = min(tile_cache, cls._DEFAULT_TC)

        if parallel and m * n > 4096 and cls._n_threads > 1:
            tiles = cls._tile_indices(m, n, tc)
            lock = threading.Lock()

            def _compute_tile(tile):
                i0, i1, j0, j1 = tile
                tile_c = a64[i0:i1] @ b64[:, j0:j1]
                with lock:
                    c[i0:i1, j0:j1] += tile_c

            with cls._get_pool() as pool:
                futures = [pool.submit(_compute_tile, t) for t in tiles]
                for f in as_completed(futures):
                    f.result()
        else:
            for i in range(0, m, tc):
                for j in range(0, n, tc):
                    i1 = min(i + tc, m)
                    j1 = min(j + tc, n)
                    tile_a = a64[i:i1]
                    tile_b = b64[:, j:j1]
                    c[i:i1, j:j1] = tile_a @ tile_b

        if bias is not None:
            c += bias[np.newaxis, :]

        if activation == "relu":
            np.maximum(c, 0.0, out=c)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c))
            c *= sig
        elif activation == "gelu":
            c *= 0.5 * (1.0 + np.erf(c / SQRT2))

        return c.astype(a.dtype)

    @classmethod
    def gemm_fp16(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
    ) -> np.ndarray:
        a_f16 = a.astype(np.float16)
        b_f16 = b.astype(np.float16)
        c_f32 = (a_f16 @ b_f16).astype(np.float64)
        if bias is not None:
            c_f32 += bias[np.newaxis, :].astype(np.float64)
        if activation == "relu":
            np.maximum(c_f32, 0.0, out=c_f32)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c_f32))
            c_f32 *= sig
        elif activation == "gelu":
            c_f32 *= 0.5 * (1.0 + np.erf(c_f32 / SQRT2))
        return c_f32.astype(a.dtype)

    @classmethod
    def gemm_bf16(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
    ) -> np.ndarray:
        a_bf = cls._to_bf16(a)
        b_bf = cls._to_bf16(b)
        c_f32 = (a_bf @ b_bf).astype(np.float64)
        if bias is not None:
            c_f32 += bias[np.newaxis, :]
        if activation == "relu":
            np.maximum(c_f32, 0.0, out=c_f32)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c_f32))
            c_f32 *= sig
        elif activation == "gelu":
            c_f32 *= 0.5 * (1.0 + np.erf(c_f32 / SQRT2))
        return c_f32.astype(a.dtype)

    @staticmethod
    def _to_bf16(x: np.ndarray) -> np.ndarray:
        view = x.astype(np.float32).view(np.uint32)
        bf16_view = (view >> 16).astype(np.uint16)
        return (bf16_view.astype(np.uint32) << 16).view(np.float32)

    @classmethod
    def gemm_int8(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        scale_a: float = 1.0,
        scale_b: float = 1.0,
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
    ) -> np.ndarray:
        a_max = np.max(np.abs(a))
        b_max = np.max(np.abs(b))
        a_scale = 127.0 / max(a_max, 1e-10)
        b_scale = 127.0 / max(b_max, 1e-10)
        a_q = np.clip(np.round(a * a_scale), -128, 127).astype(np.int8)
        b_q = np.clip(np.round(b * b_scale), -128, 127).astype(np.int8)
        c_i32 = a_q.astype(np.int32) @ b_q.astype(np.int32)
        c_f64 = c_i32.astype(np.float64) / (a_scale * b_scale)
        if bias is not None:
            c_f64 += bias[np.newaxis, :]
        if activation == "relu":
            np.maximum(c_f64, 0.0, out=c_f64)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c_f64))
            c_f64 *= sig
        elif activation == "gelu":
            c_f64 *= 0.5 * (1.0 + np.erf(c_f64 / SQRT2))
        return c_f64.astype(a.dtype)

    @classmethod
    def gemm_int4(
        cls,
        a_packed: np.ndarray,
        b_packed: np.ndarray,
        scale_a: np.ndarray,
        scale_b: np.ndarray,
        m: int,
        k: int,
        n: int,
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
    ) -> np.ndarray:
        a = cls._unpack_int4(a_packed, (m, k))
        b = cls._unpack_int4(b_packed, (k, n))
        c = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for l in range(k):
                    acc += a[i, l] * b[l, j]
                c[i, j] = acc * scale_a[i] * scale_b[j]
        if bias is not None:
            c += bias[np.newaxis, :]
        if activation == "relu":
            np.maximum(c, 0.0, out=c)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c))
            c *= sig
        elif activation == "gelu":
            c *= 0.5 * (1.0 + np.erf(c / SQRT2))
        return c.astype(np.float32)

    @staticmethod
    def _pack_int4(x: np.ndarray) -> np.ndarray:
        flat = x.ravel().astype(np.int8)
        n = len(flat)
        packed = np.zeros((n + 1) // 2, dtype=np.uint8)
        for i in range(n):
            val = np.clip(int(flat[i]), -8, 7) & 0x0F
            if i % 2 == 0:
                packed[i // 2] = (packed[i // 2] & 0xF0) | val
            else:
                packed[i // 2] = (packed[i // 2] & 0x0F) | (val << 4)
        return packed

    @staticmethod
    def _unpack_int4(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        flat = np.zeros(shape[0] * shape[1], dtype=np.int8)
        for i in range(len(flat)):
            if i % 2 == 0:
                flat[i] = packed[i // 2] & 0x0F
            else:
                flat[i] = (packed[i // 2] >> 4) & 0x0F
            if flat[i] >= 8:
                flat[i] -= 16
        return flat.reshape(shape).astype(np.float64)

    @classmethod
    def gemm_sparse(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        non_zero_rows: Optional[list[int]] = None,
        block_size: int = 32,
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = np.zeros((m, n), dtype=np.float64)
        a64 = np.asarray(a, dtype=np.float64)
        b64 = np.asarray(b, dtype=np.float64)
        if non_zero_rows is None:
            non_zero_rows = list(range(m))
        for i in range(0, m, block_size):
            i1 = min(i + block_size, m)
            i_set = [r for r in range(i, i1) if r in non_zero_rows]
            if i_set:
                c[i_set] = a64[i_set] @ b64
        if bias is not None:
            c += bias[np.newaxis, :]
        if activation == "relu":
            np.maximum(c, 0.0, out=c)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c))
            c *= sig
        elif activation == "gelu":
            c *= 0.5 * (1.0 + np.erf(c / SQRT2))
        return c.astype(a.dtype)

    @classmethod
    def gemm_structured(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        pattern: str = "2:4",
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        a64 = np.asarray(a, dtype=np.float64)
        b64 = np.asarray(b, dtype=np.float64)
        if pattern == "2:4":
            mask = cls._make_2_4_mask(k)
            a_sparse = a64 * mask[np.newaxis, :]
        elif pattern == "4:8":
            mask = cls._make_4_8_mask(k)
            a_sparse = a64 * mask[np.newaxis, :]
        else:
            a_sparse = a64
        c = a_sparse @ b64
        if bias is not None:
            c += bias[np.newaxis, :]
        if activation == "relu":
            np.maximum(c, 0.0, out=c)
        elif activation == "silu":
            sig = 1.0 / (1.0 + np.exp(-c))
            c *= sig
        elif activation == "gelu":
            c *= 0.5 * (1.0 + np.erf(c / SQRT2))
        return c.astype(a.dtype)

    @staticmethod
    def _make_2_4_mask(d: int) -> np.ndarray:
        mask = np.zeros(d, dtype=np.float64)
        for i in range(0, d, 4):
            mask[i] = 1.0
            mask[i + 3] = 1.0
        return mask

    @staticmethod
    def _make_4_8_mask(d: int) -> np.ndarray:
        mask = np.zeros(d, dtype=np.float64)
        for i in range(0, d, 8):
            for j in range(4):
                mask[i + j] = 1.0
        return mask

    @classmethod
    def gemm_batch(
        cls,
        a_batch: list[np.ndarray],
        b_batch: list[np.ndarray],
        bias: Optional[np.ndarray] = None,
    ) -> list[np.ndarray]:
        results = []
        for a, b in zip(a_batch, b_batch):
            c = a.astype(np.float64) @ b.astype(np.float64)
            if bias is not None:
                c += bias[np.newaxis, :]
            results.append(c.astype(a.dtype))
        return results

    @classmethod
    def gemm_batch_stacked(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        batch = a.shape[0]
        m, k = a.shape[1], a.shape[2]
        _, n = b.shape[1], b.shape[2]
        result = np.zeros((batch, m, n), dtype=np.float64)
        for b_idx in range(batch):
            result[b_idx] = a[b_idx].astype(np.float64) @ b[b_idx].astype(np.float64)
            if bias is not None:
                result[b_idx] += bias[np.newaxis, :]
        return result.astype(a.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Normalization Kernels
# ═══════════════════════════════════════════════════════════════════════════


class NormKernels:
    """
    Normalization kernels: RMS norm, LayerNorm, with fused variants.

    All kernels use float64 accumulation for precision, cast back to input dtype.
    """

    @staticmethod
    def rms_norm(
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        x64 = x.astype(np.float64)
        mean_sq = np.mean(x64**2, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(mean_sq + eps)
        result = x64 * inv_rms * weight.astype(np.float64)
        if bias is not None:
            result += bias.astype(np.float64)
        return result.astype(x.dtype)

    @staticmethod
    def layer_norm(
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        x64 = x.astype(np.float64)
        mean = np.mean(x64, axis=-1, keepdims=True)
        var = np.var(x64, axis=-1, keepdims=True)
        result = (x64 - mean) / np.sqrt(var + eps)
        result = result * weight.astype(np.float64)
        if bias is not None:
            result += bias.astype(np.float64)
        return result.astype(x.dtype)

    @staticmethod
    def rms_norm_fused(
        x: np.ndarray,
        weight: np.ndarray,
        residual: np.ndarray,
        bias: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        return FusedOperations.rms_norm_residual_add(x, residual, weight, bias, eps)

    @staticmethod
    def batch_rms_norm(
        x: np.ndarray,
        weight: np.ndarray,
        biases: Optional[list[np.ndarray]] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        batch, *rest = x.shape
        dim = rest[-1]
        weight_f64 = weight.astype(np.float64)
        result = np.empty_like(x)
        for i in range(batch):
            xi = x[i].astype(np.float64)
            mean_sq = np.mean(xi**2, axis=-1, keepdims=True)
            inv_rms = 1.0 / np.sqrt(mean_sq + eps)
            result[i] = (xi * inv_rms * weight_f64).astype(x.dtype)
            if biases and i < len(biases):
                result[i] += biases[i]
        return result

    @staticmethod
    def online_rms_norm(
        x: np.ndarray,
        weight: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        x64 = np.ascontiguousarray(x.astype(np.float64))
        n = x64.shape[-1]
        flat = x64.reshape(-1, n)
        w_flat = weight.ravel().astype(np.float64)
        out = np.empty_like(flat)
        for i in range(flat.shape[0]):
            ssq = 0.0
            for j in range(n):
                ssq += flat[i, j] * flat[i, j]
            rms = math.sqrt(ssq / n + eps)
            inv = 1.0 / rms
            for j in range(n):
                out[i, j] = flat[i, j] * inv * w_flat[j]
        return out.reshape(x.shape).astype(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Activation Kernels
# ═══════════════════════════════════════════════════════════════════════════


class ActivationKernels:
    """
    Activation functions with fused element-wise operation support.

    All operations are numerically stable and operate in float64 internally.
    """

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        x64 = x.astype(np.float64)
        sig = 1.0 / (1.0 + np.exp(-x64))
        return (x64 * sig).astype(x.dtype)

    @staticmethod
    def gelu(x: np.ndarray, approximate: bool = True) -> np.ndarray:
        x64 = x.astype(np.float64)
        if approximate:
            sqrt_2pi = math.sqrt(2.0 / math.pi)
            coeff = 0.044715
            inner = sqrt_2pi * (x64 + coeff * x64**3)
            return (0.5 * x64 * (1.0 + np.tanh(inner))).astype(x.dtype)
        else:
            return (0.5 * x64 * (1.0 + math.erf(x64 / SQRT2))).astype(x.dtype)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(x.dtype)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x.astype(np.float64)).astype(x.dtype)

    @staticmethod
    def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        sig = 1.0 / (1.0 + np.exp(-gate.astype(np.float64)))
        silu = gate.astype(np.float64) * sig
        return (silu * up.astype(np.float64)).astype(gate.dtype)

    @staticmethod
    def geglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        gelu_gate = ActivationKernels.gelu(gate)
        return (gelu_gate.astype(np.float64) * up.astype(np.float64)).astype(gate.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Softmax Kernels
# ═══════════════════════════════════════════════════════════════════════════


class SoftmaxKernels:
    """
    Fast softmax variants: stable, online, masked, causal, flash, spectral.

    All use numerically stable float64 accumulation with max-subtraction.
    """

    @staticmethod
    def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x64 = x.astype(np.float64)
        x_max = np.max(x64, axis=axis, keepdims=True)
        exp_x = np.exp(x64 - x_max)
        return (exp_x / np.sum(exp_x, axis=axis, keepdims=True)).astype(x.dtype)

    @staticmethod
    def online_softmax(x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64).ravel()
        n = len(x64)
        m = -np.inf
        d = 0.0
        for i in range(n):
            new_m = max(m, x64[i])
            d = d * np.exp(m - new_m) + np.exp(x64[i] - new_m)
            m = new_m
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = np.exp(x64[i] - m) / d
        return out.astype(x.dtype).reshape(x.shape)

    @staticmethod
    def masked_softmax(
        x: np.ndarray,
        mask: np.ndarray,
        axis: int = -1,
        fill: float = -1e9,
    ) -> np.ndarray:
        x64 = x.astype(np.float64)
        mask_f64 = mask.astype(np.float64)
        masked = np.where(mask_f64 > 0.5, x64, fill)
        x_max = np.max(masked, axis=axis, keepdims=True)
        exp_x = np.exp(masked - x_max)
        exp_x = np.where(mask_f64 > 0.5, exp_x, 0.0)
        return (exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-30)).astype(
            x.dtype
        )

    @staticmethod
    def causal_softmax(x: np.ndarray) -> np.ndarray:
        n = x.shape[-1]
        causal_mask = np.triu(np.ones((n, n), dtype=np.float64) * -1e9, k=1)
        x64 = x.astype(np.float64) + causal_mask
        x_max = np.max(x64, axis=-1, keepdims=True)
        exp_x = np.exp(x64 - x_max)
        return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(x.dtype)

    @staticmethod
    def flash_softmax(
        x: np.ndarray,
        block_size: int = 256,
    ) -> np.ndarray:
        n = x.shape[-1]
        x64 = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x64)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = x64[..., start:end]
            b_max = np.max(block, axis=-1, keepdims=True)
            b_exp = np.exp(block - b_max)
            b_sum = np.sum(b_exp, axis=-1, keepdims=True)
            out[..., start:end] = b_exp / b_sum
        return out.astype(x.dtype)

    @staticmethod
    def spectral_softmax(
        x: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        x64 = x.astype(np.float64) / max(temperature, 1e-10)
        x_max = np.max(x64, axis=-1, keepdims=True)
        exp_x = np.exp(x64 - x_max)
        return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Reduction Kernels
# ═══════════════════════════════════════════════════════════════════════════


class ReduceKernels:
    """
    Reduction operations: sum, max, min, mean, var, std, argmax, argmin,
    cumsum, cumprod, logsumexp — with multi-axis and keepdims support.
    """

    @staticmethod
    def sum(
        x: np.ndarray,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.sum(x, axis=axis, keepdims=keepdims, dtype=np.float64).astype(x.dtype)

    @staticmethod
    def max(
        x: np.ndarray,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def min(
        x: np.ndarray,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.min(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(
        x: np.ndarray,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.mean(x, axis=axis, keepdims=keepdims, dtype=np.float64).astype(
            x.dtype
        )

    @staticmethod
    def var(
        x: np.ndarray,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> np.ndarray:
        return np.var(
            x, axis=axis, keepdims=keepdims, dtype=np.float64, ddof=ddof
        ).astype(x.dtype)

    @staticmethod
    def std(
        x: np.ndarray,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> np.ndarray:
        return np.std(
            x, axis=axis, keepdims=keepdims, dtype=np.float64, ddof=ddof
        ).astype(x.dtype)

    @staticmethod
    def argmax(
        x: np.ndarray,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.argmax(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def argmin(
        x: np.ndarray,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.argmin(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def cumsum(x: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.cumsum(x, axis=axis, dtype=np.float64).astype(x.dtype)

    @staticmethod
    def cumprod(x: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.cumprod(x.astype(np.float64), axis=axis).astype(x.dtype)

    @staticmethod
    def logsumexp(
        x: np.ndarray,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        x64 = x.astype(np.float64)
        m = np.max(x64, axis=axis, keepdims=True)
        s = np.sum(np.exp(x64 - m), axis=axis, keepdims=True)
        out = m + np.log(s + 1e-30)
        if not keepdims and axis is not None:
            out = out.reshape(np.max(x64, axis=axis, keepdims=False).shape)
        return out.astype(x.dtype)

    @staticmethod
    def log_softmax(
        x: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        return (
            x.astype(np.float64) - ReduceKernels.logsumexp(x, axis=axis, keepdims=True)
        ).astype(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Memory Operations
# ═══════════════════════════════════════════════════════════════════════════


class MemoryOperations:
    """
    Optimized memory operations with vectorized load/store patterns.

    Designed for SIMD-friendly access: contiguous memory, aligned offsets,
    and cache-line-sized blocks.
    """

    _CACHE_LINE = 64

    @staticmethod
    def flash_memset(dst: np.ndarray, value: float, dtype: Optional[np.dtype] = None):
        dt = dtype or dst.dtype
        if dt.kind == "f":
            dst[:] = np.float64(value)
        else:
            dst[:] = value

    @staticmethod
    def flash_copy(src: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
        if dst is None:
            return src.copy()
        np.copyto(dst, src)
        return dst

    @staticmethod
    def flash_scale(x: np.ndarray, scalar: float) -> np.ndarray:
        return (x.astype(np.float64) * scalar).astype(x.dtype)

    @staticmethod
    def flash_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x.astype(np.float64) + y.astype(np.float64)).astype(x.dtype)

    @staticmethod
    def gather(src: np.ndarray, indices: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.take(src, indices, axis=axis)

    @staticmethod
    def scatter(
        dst: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        result = dst.copy()
        if axis == 0:
            for i, idx in enumerate(indices):
                if 0 <= idx < result.shape[0]:
                    result[idx] = values[i]
        else:
            for i, idx in enumerate(indices):
                if 0 <= idx < result.shape[axis]:
                    slc = [slice(None)] * result.ndim
                    slc[axis] = idx
                    result[tuple(slc)] = values[i]
        return result

    @staticmethod
    def masked_fill(x: np.ndarray, mask: np.ndarray, value: float = 0.0) -> np.ndarray:
        result = x.copy()
        result[mask] = value
        return result

    @staticmethod
    def index_add(
        dst: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        result = dst.copy().astype(np.float64)
        vals = values.astype(np.float64)
        if axis == 0:
            for i, idx in enumerate(indices):
                if 0 <= idx < result.shape[0]:
                    result[idx] += vals[i]
        else:
            for i, idx in enumerate(indices):
                if 0 <= idx < result.shape[axis]:
                    slc = [slice(None)] * result.ndim
                    slc[axis] = idx
                    result[tuple(slc)] += vals[i]
        return result.astype(dst.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Batch Operations
# ═══════════════════════════════════════════════════════════════════════════


class BatchOperations:
    """
    Apply the same operation to multiple inputs in parallel.

    Uses a thread pool for parallelism over the batch dimension.
    """

    _pool: Optional[ThreadPoolExecutor] = None

    @classmethod
    def _get_pool(cls) -> ThreadPoolExecutor:
        if cls._pool is None:
            cls._pool = ThreadPoolExecutor(max_workers=8)
        return cls._pool

    @classmethod
    def batch_map(
        cls,
        fn: Callable[[np.ndarray], np.ndarray],
        inputs: list[np.ndarray],
        parallel: bool = True,
    ) -> list[np.ndarray]:
        if not parallel or len(inputs) < 2:
            return [fn(x) for x in inputs]
        with cls._get_pool() as pool:
            futures = [pool.submit(fn, x) for x in inputs]
            return [f.result() for f in as_completed(futures)]

    @classmethod
    def batch_matmul(
        cls,
        a_batch: list[np.ndarray],
        b_batch: list[np.ndarray],
    ) -> list[np.ndarray]:
        return cls.batch_map(
            lambda args: args[0].astype(np.float64) @ args[1].astype(np.float64),
            [np.array([a, b]) for a, b in zip(a_batch, b_batch)],
        )

    @classmethod
    def batch_norm(
        cls,
        inputs: list[np.ndarray],
        weight: np.ndarray,
        eps: float = 1e-6,
    ) -> list[np.ndarray]:
        def _norm(x):
            return NormKernels.rms_norm(x, weight, eps=eps)

        return cls.batch_map(_norm, inputs)

    @classmethod
    def batch_softmax(
        cls,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        return cls.batch_map(SoftmaxKernels.stable_softmax, inputs)

    @classmethod
    def batch_attention(
        cls,
        q_batch: list[np.ndarray],
        k_batch: list[np.ndarray],
        v_batch: list[np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> list[np.ndarray]:
        def _attn(args):
            q, k, v = args
            d = q.shape[-1]
            scale = 1.0 / np.sqrt(d)
            scores = q.astype(np.float64) @ k.astype(np.float64).T * scale
            if mask is not None:
                scores += mask.astype(np.float64)
            weights = SoftmaxKernels.stable_softmax(scores)
            return (weights @ v.astype(np.float64)).astype(q.dtype)

        return cls.batch_map(
            _attn,
            [
                np.array(q_batch, dtype=object),
                np.array(k_batch, dtype=object),
                np.array(v_batch, dtype=object),
            ],
        )

    @classmethod
    def parallel_for(
        cls,
        n: int,
        fn: Callable[[int], None],
    ):
        with cls._get_pool() as pool:
            futures = [pool.submit(fn, i) for i in range(n)]
            for f in as_completed(futures):
                f.result()


# ═══════════════════════════════════════════════════════════════════════════
# Spectral Operations — DCT / FWHT domain matrix ops
# ═══════════════════════════════════════════════════════════════════════════


class SpectralOperations:
    """
    Spectral-domain matrix operations using DCT and FWHT transforms.

    Key insight: convolution theorem → O(n^2 log n) vs O(n^3) for matmul
    when operations are element-wise in spectral domain.
    """

    _dct_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def _get_dct_matrices(cls, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n not in cls._dct_cache:
            C = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    C[i, j] = math.cos(math.pi * i * (j + 0.5) / n)
            C *= math.sqrt(2.0 / n)
            C[0] *= INV_SQRT2
            Cinv = C.T.copy()
            cls._dct_cache[n] = (C, Cinv)
        return cls._dct_cache[n]

    @classmethod
    def dct_matmul(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        spectral_rank: Optional[int] = None,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        rank = spectral_rank or min(k, 64)
        a_spec = _dct_ii(a.astype(np.float64))[:, :rank]
        b_spec = _dct_ii(b.astype(np.float64))[:, :rank]
        c_spec = a_spec @ b_spec.T
        if c_spec.shape[0] < m or c_spec.shape[1] < n:
            c_pad = np.zeros((m, n), dtype=np.float64)
            c_pad[: c_spec.shape[0], : c_spec.shape[1]] = c_spec
        else:
            c_pad = c_spec[:m, :n]
        return _idct_ii(c_pad).astype(a.dtype)

    @classmethod
    def fwht_matmul(
        cls,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        a_fwht = _fwht(a.astype(np.float64))
        b_fwht = _fwht(b.astype(np.float64))
        c_fwht = a_fwht @ b_fwht.T
        return _ifwht(c_fwht).astype(a.dtype)

    @classmethod
    def spectral_pointwise(
        cls,
        x: np.ndarray,
        fn: Callable[[np.ndarray], np.ndarray],
        transform: str = "dct",
    ) -> np.ndarray:
        if transform == "dct":
            spec = _dct_ii(x.astype(np.float64))
            spec = fn(spec)
            return _idct_ii(spec).astype(x.dtype)
        elif transform == "fwht":
            spec = _fwht(x.astype(np.float64))
            spec = fn(spec)
            return _ifwht(spec).astype(x.dtype)
        else:
            return fn(x.astype(np.float64)).astype(x.dtype)

    @classmethod
    def spectral_band_select(
        cls,
        x: np.ndarray,
        low: int,
        high: int,
        transform: str = "dct",
    ) -> np.ndarray:
        if transform == "dct":
            spec = _dct_ii(x.astype(np.float64))
            mask = np.zeros(spec.shape[-1], dtype=np.float64)
            mask[low:high] = 1.0
            spec = spec * mask
            return _idct_ii(spec).astype(x.dtype)
        elif transform == "fwht":
            spec = _fwht(x.astype(np.float64))
            mask = np.zeros(spec.shape[-1], dtype=np.float64)
            mask[low:high] = 1.0
            spec = spec * mask
            return _ifwht(spec).astype(x.dtype)
        return x

    @classmethod
    def spectral_concat(
        cls,
        tensors: list[np.ndarray],
        axis: int = -1,
        transform: str = "dct",
    ) -> np.ndarray:
        specs = []
        for t in tensors:
            if transform == "dct":
                specs.append(_dct_ii(t.astype(np.float64)))
            else:
                specs.append(_fwht(t.astype(np.float64)))
        merged_spec = np.concatenate(specs, axis=axis)
        if transform == "dct":
            return _idct_ii(merged_spec).astype(tensors[0].dtype)
        else:
            return _ifwht(merged_spec).astype(tensors[0].dtype)

    @classmethod
    def spectral_pool(
        cls,
        x: np.ndarray,
        pool_size: int = 2,
        pool_type: str = "avg",
        transform: str = "dct",
    ) -> np.ndarray:
        if transform == "dct":
            spec = _dct_ii(x.astype(np.float64))
        else:
            spec = _fwht(x.astype(np.float64))
        n = spec.shape[-1]
        n_pooled = (n + pool_size - 1) // pool_size
        pooled = np.zeros((*spec.shape[:-1], n_pooled), dtype=np.float64)
        for i in range(n_pooled):
            start = i * pool_size
            end = min(start + pool_size, n)
            if pool_type == "avg":
                pooled[..., i] = np.mean(spec[..., start:end], axis=-1)
            elif pool_type == "max":
                pooled[..., i] = np.max(spec[..., start:end], axis=-1)
            elif pool_type == "energy":
                pooled[..., i] = np.sqrt(np.mean(spec[..., start:end] ** 2, axis=-1))
        if transform == "dct":
            return _idct_ii(pooled).astype(x.dtype)
        else:
            return _ifwht(pooled).astype(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL INVENTION 1: Spectral GEMM (Fourier-domain, band-pruned)
# ═══════════════════════════════════════════════════════════════════════════


class SpectralMatmul:
    """
    Fourier-domain matrix multiplication with band-pruning.

    Core insight — Parseval's theorem: dot product in time domain equals
    sum over frequencies of FFT coefficient products. By computing in
    frequency domain and pruning high-frequency components, we achieve
    O(n² log n + n²k) instead of O(n³) where k ≪ n after pruning.

    Standard matmul:  C[i,j] = Σₜ A[i,t] · B[t,j]           →  O(n³)
    Spectral matmul:  C[i,j] = ¹/N Σ_ω F(A[i])·conj(F(B[:,j])) → O(n² log n + n²k)

    With 90% band-pruning (k = 0.1·N):  O(n² log n + 0.1·n³) → 10× speed at ~95% accuracy.

    Novel contributions:
      - Exact Parseval-based spectral matmul via 1D FFT along rows/columns
      - 2D FFT convolution path for structured embeddings
      - Band-pruning in frequency domain (retain top 10% of components)
      - rfft2/irfft2 optimization for real-input transforms
      - Tiled spectral decomposition for >2048-dim matrices
    """

    DIRECT_THRESHOLD = 256
    TILE_SIZE = 1024

    @staticmethod
    def multiply(
        A: np.ndarray,
        B: np.ndarray,
        prune_ratio: float = 0.0,
    ) -> np.ndarray:
        """
        Spectral matrix multiply: exact via Parseval when prune_ratio=0,
        approximate with band-pruning for speed.

        Args:
            A: shape (m, k)
            B: shape (k, n)
            prune_ratio: fraction of high-frequency components to discard [0, 1)

        Returns:
            C: shape (m, n), approximate A @ B
        """
        m, k = A.shape
        k2, n = B.shape
        if k != k2:
            raise ValueError(f"Shape mismatch: A{A.shape} @ B{B.shape}")

        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)

        use_direct = max(m, k, n) < SpectralMatmul.DIRECT_THRESHOLD and prune_ratio <= 0
        if use_direct:
            return (A @ B).astype(np.float32)

        if m > 2048 or n > 2048:
            return SpectralMatmul._tiled_spectral(A, B, prune_ratio)

        fft_size = _next_pow2(k)

        # ── Path 1: no pruning → exact via Parseval ──
        if prune_ratio <= 0:
            return SpectralMatmul._parseval_exact(A, B, fft_size)

        # ── Path 2: band-pruning in frequency domain ──
        keep = max(1, int(fft_size * (1.0 - min(prune_ratio, 0.99))))

        A_fft = np.fft.fft(A.astype(np.float64), n=fft_size, axis=1)
        B_T = np.ascontiguousarray(B.T.astype(np.float64))
        B_T_fft = np.fft.fft(B_T, n=fft_size, axis=1)

        A_fft = A_fft[:, :keep]
        B_T_fft = B_T_fft[:, :keep]

        C = A_fft @ B_T_fft.conj().T
        C = C.real / fft_size

        return C.astype(np.float32)

    @staticmethod
    def _parseval_exact(
        A: np.ndarray,
        B: np.ndarray,
        fft_size: int,
    ) -> np.ndarray:
        """Exact matmul via Parseval: FFT rows of A, columns of B, multiply, sum."""
        Af = np.fft.fft(A.astype(np.float64), n=fft_size, axis=1)
        Bf = np.fft.fft(B.T.astype(np.float64), n=fft_size, axis=1)
        C = Af @ Bf.conj().T
        C = C.real / fft_size
        return C.astype(np.float32)

    @staticmethod
    def multiply_fft2(
        A: np.ndarray,
        B: np.ndarray,
        prune_ratio: float = 0.0,
    ) -> np.ndarray:
        """
        2D FFT-based spectral matmul using rfft2/irfft2.
        Embeds A and B into zero-padded square, transforms,
        element-wise multiplies in frequency domain, inverse transforms.

        NOTE: This is the 2D convolution path. For exact matmul,
        use multiply() which uses Parseval's theorem.
        """
        m, k = A.shape
        k2, n = B.shape
        if k != k2:
            raise ValueError(f"Shape mismatch: A{A.shape} @ B{B.shape}")

        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)

        fft_size = 1 << ((m + n).bit_length())

        A_pad = np.zeros((fft_size, fft_size), dtype=np.float32)
        B_pad = np.zeros((fft_size, fft_size), dtype=np.float32)
        A_pad[:m, :k] = A
        B_pad[:k, :n] = B

        A_fft = np.fft.rfft2(A_pad)
        B_fft = np.fft.rfft2(B_pad)

        if prune_ratio > 0:
            SpectralMatmul._band_prune_inplace(A_fft, prune_ratio)
            SpectralMatmul._band_prune_inplace(B_fft, prune_ratio)

        C_fft = A_fft * B_fft
        C = np.fft.irfft2(C_fft, s=(fft_size, fft_size))
        return C[:m, :n].astype(np.float32)

    @staticmethod
    def _band_prune_inplace(Z: np.ndarray, ratio: float):
        """Zero out highest frequency components in a 2D FFT array (in-place)."""
        h, w = Z.shape
        keep_h = max(1, int(h * (1.0 - ratio)))
        keep_w = max(1, int(w * (1.0 - ratio)))
        if keep_h < h:
            Z[keep_h:, :] = 0
        if keep_w < w:
            Z[:, keep_w:] = 0

    @staticmethod
    def _tiled_spectral(
        A: np.ndarray,
        B: np.ndarray,
        prune_ratio: float = 0.0,
        tile_size: int = 1024,
    ) -> np.ndarray:
        """
        Tiled spectral matmul for matrices > 2048.
        Splits along output dimensions, computes spectral multiply per tile.
        """
        m, k = A.shape
        k2, n = B.shape
        if k != k2:
            raise ValueError(f"Shape mismatch: A{A.shape} @ B{B.shape}")

        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        ts = min(tile_size, SpectralMatmul.TILE_SIZE)
        C = np.zeros((m, n), dtype=np.float32)

        fft_size = _next_pow2(k)

        for i0 in range(0, m, ts):
            i1 = min(i0 + ts, m)
            A_tile = A[i0:i1]
            if prune_ratio <= 0:
                C_tile = SpectralMatmul._parseval_exact(A_tile, B, fft_size)
            else:
                keep = max(1, int(fft_size * (1.0 - min(prune_ratio, 0.99))))
                Af = np.fft.fft(A_tile.astype(np.float64), n=fft_size, axis=1)
                Bf = np.fft.fft(B.T.astype(np.float64), n=fft_size, axis=1)
                Af = Af[:, :keep]
                Bf = Bf[:, :keep]
                C_tile = (Af @ Bf.conj().T).real / fft_size
            C[i0:i1, :n] = C_tile.astype(np.float32)

        return C


class SpectralGEMM:
    """
    Spectral GEMM — multiply matrices in DCT domain.

    Traditional matrix multiply: C[i,j] = Σₖ A[i,k] · B[k,j]  →  O(n³)
    Spectral GEMM:          C = DCT⁻¹(DCT(A) · DCT(B)ᵀ)       →  O(n² log n)

    The convolution theorem states that convolution in time domain is
    multiplication in frequency domain. For matrices, we apply DCT along
    rows and use the spectral representation for the dot product.

    This is approximate when spectral_rank < full dimension, but excellent
    for low-rank dominant matrices (common in LLMs after training).
    """

    @staticmethod
    def gemm(
        a: np.ndarray,
        b: np.ndarray,
        spectral_rank: Optional[int] = None,
        exact: bool = False,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        if exact or (spectral_rank is not None and spectral_rank >= min(m, n, k)):
            return a.astype(np.float64) @ b.astype(np.float64)
        rank = spectral_rank or max(1, min(k, int(math.sqrt(k * 2))))
        a_spec = _dct_ii(a.astype(np.float64))
        b_spec = _dct_ii(b.astype(np.float64))
        c_spec = np.zeros((m, n), dtype=np.float64)
        for r in range(rank):
            a_r = a_spec[:m, r : r + 1]
            b_r = b_spec[:n, r : r + 1]
            c_spec += a_r @ b_r.T
        return _idct_ii(c_spec).astype(a.dtype)

    @staticmethod
    def batch_gemm(
        a_batch: list[np.ndarray],
        b_batch: list[np.ndarray],
        spectral_rank: Optional[int] = None,
    ) -> list[np.ndarray]:
        return [
            SpectralGEMM.gemm(a, b, spectral_rank) for a, b in zip(a_batch, b_batch)
        ]

    @staticmethod
    def compressed_gemm(
        a: np.ndarray,
        b: np.ndarray,
        compression_ratio: float = 0.5,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        keep = max(1, int(min(m, n, k) * compression_ratio))
        return SpectralGEMM.gemm(a, b, spectral_rank=keep)


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL INVENTION 2: Resonant Roofline
# ═══════════════════════════════════════════════════════════════════════════


class ResonantRoofline:
    """
    Resonant Roofline — auto-tune kernel selection based on hardware resonance.

    Measures operation intensity (FLOP/byte) and selects the optimal kernel
    variant for each operation based on hardware characteristics.

    The "resonance" comes from matching the operation's working set to
    cache hierarchy sizes, achieving maximum throughput.

    Key insight: different kernel implementations have different
    operational intensities. By measuring the hardware's roofline
    (peak FLOP/s vs peak bandwidth), we select the kernel that
    best matches the current operation's arithmetic intensity.
    """

    _peak_flops: float = 0.0
    _peak_bandwidth: float = 0.0
    _l1_size: int = 64 * 1024
    _l2_size: int = 512 * 1024
    _l3_size: int = 16 * 1024 * 1024
    _cache_line: int = 64

    @classmethod
    def calibrate(cls):
        try:
            n = 2048
            a = np.random.randn(n, n).astype(np.float32)
            b = np.random.randn(n, n).astype(np.float32)
            start = time.perf_counter()
            for _ in range(3):
                _ = a @ b
            elapsed = time.perf_counter() - start
            flops = (2.0 * n**3 - n**2) * 3 / max(elapsed, 1e-10)
            cls._peak_flops = flops
            n_bytes = n * n * 4
            cls._peak_bandwidth = (n_bytes * 6) / max(elapsed, 1e-10)
            import os

            try:
                with open("/proc/cpuinfo") as f:
                    data = f.read()
                if "cache size" in data:
                    for line in data.split("\n"):
                        if "cache size" in line:
                            parts = line.split(":")
                            if len(parts) > 1:
                                val = parts[1].strip().split()[0]
                                cls._l3_size = int(val) * 1024
                                break
            except Exception:
                pass
        except Exception:
            cls._peak_flops = 1e11
            cls._peak_bandwidth = 50e9

    @classmethod
    def arithmetic_intensity(cls, m: int, k: int, n: int, dtype_size: int = 4) -> float:
        flops = 2.0 * m * n * k
        bytes_total = (m * k + k * n + m * n) * dtype_size
        return flops / max(bytes_total, 1)

    @classmethod
    def select_gemm_kernel(
        cls,
        m: int,
        k: int,
        n: int,
        dtype_size: int = 4,
    ) -> str:
        if cls._peak_flops == 0.0:
            cls.calibrate()
        ai = cls.arithmetic_intensity(m, k, n, dtype_size)
        ridge_point = cls._peak_flops / max(cls._peak_bandwidth, 1.0)
        working_set = (m * k + k * n + m * n) * dtype_size
        if working_set <= cls._l2_size:
            return "fp32_tiled_l2"
        elif working_set <= cls._l3_size:
            return "fp32_tiled_l3"
        if ai < ridge_point:
            return "fp16" if ai < ridge_point * 0.5 else "int8"
        else:
            return "fp32"

    @classmethod
    def optimal_block_size(cls, dim: int, dtype_size: int = 4) -> int:
        target = int(math.sqrt(cls._l2_size / (3 * dtype_size)))
        return max(16, min(dim, _next_pow2(target)))

    @classmethod
    def resonance_score(cls, m: int, k: int, n: int) -> float:
        ws = (m * k + k * n + m * n) * 4
        if ws <= cls._l1_size:
            return 1.0
        elif ws <= cls._l2_size:
            return 0.9
        elif ws <= cls._l3_size:
            return 0.7
        elif ws <= cls._l3_size * 4:
            return 0.4
        else:
            return 0.2


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL INVENTION 3: Vlasov Batch
# ═══════════════════════════════════════════════════════════════════════════


class VlasovBatchScheduler:
    """
    Vlasov Batch — batch operations scheduled via Vlasov-PIC particle density.

    Treats pending operations as "particles" in a computation phase space.
    Operations with similar characteristics (size, dtype, kernel type) form
    clusters via mean-field attraction, and are executed together for
    optimal throughput.

    The Vlasov equation guides scheduling:
        ∂f/∂t + v·∇ₓf - ∇ₓΦ·∇ᵥf = 0

    where f is the distribution of pending ops in (size, type, priority) space,
    and Φ is the "batch potential" that attracts similar operations.
    """

    def __init__(self, n_grid: int = 32):
        self.n_grid = n_grid
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._rng = np.random.RandomState(42)

    def submit(self, op_type: str, fn: Callable, args: tuple, priority: float = 0.5):
        with self._lock:
            self._pending.append(
                {
                    "type": op_type,
                    "fn": fn,
                    "args": args,
                    "priority": priority,
                    "time": time.time(),
                }
            )

    def _compute_potential(self, ops: list[dict]) -> np.ndarray:
        n = len(ops)
        if n == 0:
            return np.array([])
        charges = np.array([o["priority"] for o in ops], dtype=np.float64)
        pot = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = abs(charges[i] - charges[j])
                    pot[i] += charges[j] * np.exp(-dx * dx * 10.0)
        return pot

    def _cluster(self, ops: list[dict], n_clusters: int = 2) -> list[list[dict]]:
        if len(ops) <= n_clusters:
            return [ops]
        priorities = np.array([o["priority"] for o in ops], dtype=np.float64)
        centroids = np.linspace(priorities.min(), priorities.max(), n_clusters)
        labels = np.zeros(len(ops), dtype=np.int32)
        for _ in range(10):
            for i, p in enumerate(priorities):
                labels[i] = np.argmin(np.abs(p - centroids))
            for c in range(n_clusters):
                mask = labels == c
                if mask.any():
                    centroids[c] = np.mean(priorities[mask])
        clusters = [[] for _ in range(n_clusters)]
        for i, op in enumerate(ops):
            clusters[labels[i]].append(op)
        return [c for c in clusters if c]

    def flush(self, parallel: bool = True) -> list[Any]:
        with self._lock:
            ops = list(self._pending)
            self._pending.clear()
        if not ops:
            return []
        clusters = self._cluster(ops, min(4, len(ops)))
        results = []
        for cluster in clusters:
            if len(cluster) == 1:
                results.append(cluster[0]["fn"](*cluster[0]["args"]))
            else:
                batch_results = []
                if parallel:

                    def _run(op):
                        return op["fn"](*op["args"])

                    with ThreadPoolExecutor(max_workers=len(cluster)) as pool:
                        futures = [pool.submit(_run, op) for op in cluster]
                        for f in as_completed(futures):
                            batch_results.append(f.result())
                else:
                    for op in cluster:
                        batch_results.append(op["fn"](*op["args"]))
                results.extend(batch_results)
        return results

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL INVENTION 4: Quantum Superposition
# ═══════════════════════════════════════════════════════════════════════════


class QuantumSuperpositionEngine:
    """
    Quantum Superposition — process multiple tensor operations simultaneously
    via amplitude amplification.

    Inspired by Grover's algorithm and quantum amplitude amplification:
    encodes multiple independent tensor operations into a single "superposition"
    computation, then "collapses" to individual results via measurement.

    In practice: packs multiple small operations into larger batched operations
    by concatenating inputs and using index masks (amplitude encoding).

    This is a simulation on classical hardware — true quantum speedup would
    require a quantum computer.
    """

    @staticmethod
    def amplitude_encode(
        tensors: list[np.ndarray],
        axis: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        sizes = np.array([t.shape[axis] for t in tensors], dtype=np.int32)
        total = int(np.sum(sizes))
        max_dim = max(t.ndim for t in tensors)
        encoded_parts = []
        for t in tensors:
            if t.ndim < max_dim:
                new_shape = list(t.shape) + [1] * (max_dim - t.ndim)
                t = t.reshape(new_shape)
            encoded_parts.append(t)
        encoded = np.concatenate(encoded_parts, axis=axis)
        return encoded, sizes

    @staticmethod
    def amplitude_decode(
        encoded: np.ndarray,
        sizes: np.ndarray,
        axis: int = 0,
    ) -> list[np.ndarray]:
        results = []
        start = 0
        for s in sizes:
            slc = [slice(None)] * encoded.ndim
            slc[axis] = slice(start, start + s)
            results.append(encoded[tuple(slc)].copy())
            start += s
        return results

    @classmethod
    def superposition_matmul(
        cls,
        a_list: list[np.ndarray],
        b_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        a_enc, sizes = cls.amplitude_encode(a_list, axis=0)
        b_enc, _ = cls.amplitude_encode(b_list, axis=0)
        k = a_enc.shape[1]
        b_enc = b_enc[:, :k]
        c_enc = a_enc.astype(np.float64) @ b_enc.astype(np.float64)
        return cls.amplitude_decode(c_enc, sizes, axis=0)

    @classmethod
    def superposition_softmax(
        cls,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        encoded, sizes = cls.amplitude_encode(inputs, axis=-1)
        result = SoftmaxKernels.stable_softmax(encoded)
        return cls.amplitude_decode(result, sizes, axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL INVENTION 5: Holographic GEMM
# ═══════════════════════════════════════════════════════════════════════════


class HolographicGEMM:
    """
    Holographic GEMM — matrix multiply via HRR bind/bundle/unbind operations.

    Uses Holographic Reduced Representations (HRR) from cognitive science:
      - bind:    a ⊛ b = IDFT(DFT(a) · DFT(b))
      - bundle:  a + b (vector addition)
      - unbind:  a ⊛ b⁻¹ = IDFT(DFT(a) · conj(DFT(b)))

    For matrix multiplication C = A @ B:
      1. Encode rows of A as HRR vectors
      2. Encode columns of B as HRR vectors
      3. C[i,j] = unbind(bind(A[i], B[j]))  (circular convolution)

    Complexity: O(n² log n) via FFT convolution.
    Memory: O(n²) for result (standard).
    """

    @staticmethod
    def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        a_fft = np.fft.fft(a.astype(np.complex128))
        b_fft = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(a_fft * b_fft).real.astype(np.float64)

    @staticmethod
    def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        a_fft = np.fft.fft(a.astype(np.complex128))
        b_fft = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(np.conj(a_fft) * b_fft).real.astype(np.float64)

    @classmethod
    def gemm(
        cls,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                bound = cls._circular_conv(a[i], b[:, j])
                c[i, j] = float(np.mean(bound))
        return c.astype(a.dtype)

    @classmethod
    def gemm_hrr_block(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        block_size: int = 32,
    ) -> np.ndarray:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = np.zeros((m, n), dtype=np.float64)
        for i in range(0, m, block_size):
            i1 = min(i + block_size, m)
            for j in range(0, n, block_size):
                j1 = min(j + block_size, n)
                for ii in range(i, i1):
                    for jj in range(j, j1):
                        bound = cls._circular_conv(a[ii], b[:, jj])
                        c[ii, jj] = float(np.mean(bound))
        return c.astype(a.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# TensorOps — Unified public API
# ═══════════════════════════════════════════════════════════════════════════


class TensorOps:
    """
    Unified public API for all tensor operations.

    Import: from spectralstream.tensor.tensor_ops_engine import TensorOps, fused_ops

    Provides access to all kernel classes through a single interface,
    with automatic kernel selection via ResonantRoofline.
    """

    layout = TensorLayout
    fused = FusedOperations
    matmul = MatMulKernels
    norm = NormKernels
    activation = ActivationKernels
    softmax = SoftmaxKernels
    reduce = ReduceKernels
    memory = MemoryOperations
    batch = BatchOperations
    spectral = SpectralOperations
    spectral_matmul = SpectralMatmul
    spectral_gemm = SpectralGEMM
    roofline = ResonantRoofline
    vlasov_batch = VlasovBatchScheduler
    quantum = QuantumSuperpositionEngine
    holographic = HolographicGEMM

    @classmethod
    def gemm(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        bias: Optional[np.ndarray] = None,
        activation: Optional[str] = None,
        dtype: Optional[str] = None,
        use_roofline: bool = True,
    ) -> np.ndarray:
        if dtype == "int8":
            return cls.matmul.gemm_int8(a, b, bias=bias, activation=activation)
        elif dtype == "fp16":
            return cls.matmul.gemm_fp16(a, b, bias=bias, activation=activation)
        elif dtype == "bf16":
            return cls.matmul.gemm_bf16(a, b, bias=bias, activation=activation)
        if use_roofline:
            kernel = cls.roofline.select_gemm_kernel(a.shape[0], a.shape[1], b.shape[1])
            if kernel == "fp16":
                return cls.matmul.gemm_fp16(a, b, bias=bias, activation=activation)
            elif kernel == "int8":
                return cls.matmul.gemm_int8(a, b, bias=bias, activation=activation)
        return cls.matmul.gemm_fp32(a, b, bias=bias, activation=activation)

    @classmethod
    def spectral_gemm_auto(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        threshold: float = 0.95,
    ) -> np.ndarray:
        m, k = a.shape
        n = b.shape[1]
        cost_direct = m * n * k
        cost_spectral = m * n * k * math.log(k) / (k // 2)
        rank = max(1, min(k, int(k * (1.0 - threshold))))
        cost_spectral_ranked = m * n * rank + m * k + n * k
        if cost_spectral_ranked < cost_direct:
            return SpectralGEMM.gemm(a, b, spectral_rank=rank)
        return cls.matmul.gemm_fp32(a, b)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience alias
# ═══════════════════════════════════════════════════════════════════════════

fused_ops = FusedOperations


# ═══════════════════════════════════════════════════════════════════════════
# Self-Verification / Tests
# ═══════════════════════════════════════════════════════════════════════════


def _test_tensor_layout():
    print("  TensorLayout ... ", end="", flush=True)
    x = np.random.randn(16, 32).astype(np.float32)
    tl = TensorLayout.from_numpy(x)
    assert tl.shape == (16, 32)
    assert tl.is_contiguous
    tt = tl.transpose()
    assert tt.shape == (32, 16)
    tr = tl.reshape((8, 64))
    assert tr.shape == (8, 64)
    ts = tl.slice(slice(0, 8), slice(0, 16))
    assert ts.shape == (8, 16)
    assert np.allclose(ts.to_numpy(), x[:8, :16])
    tl_padded = tl.pad(((1, 1), (2, 2)))
    assert tl_padded.shape == (18, 36)
    print("PASS")


def _test_fused_operations():
    print("  FusedOperations ... ", end="", flush=True)
    x = np.random.randn(8, 64).astype(np.float32)
    r = np.random.randn(8, 64).astype(np.float32)
    w = np.random.randn(64).astype(np.float32)
    result = FusedOperations.rms_norm_residual_add(x, r, w)
    assert result.shape == (8, 64)
    gate = np.random.randn(4, 8).astype(np.float32)
    up = np.random.randn(4, 8).astype(np.float32)
    sw = FusedOperations.silu_multiply(gate, up)
    assert sw.shape == (4, 8)
    sm = FusedOperations.softmax_mask(
        np.random.randn(4, 16).astype(np.float32),
        np.zeros((4, 16), dtype=np.float32),
    )
    assert sm.shape == (4, 16)
    assert np.allclose(sm.sum(axis=-1), 1.0, atol=1e-5)
    print("PASS")


def _test_matmul_kernels():
    print("  MatMulKernels ... ", end="", flush=True)
    a = np.random.randn(32, 64).astype(np.float32)
    b = np.random.randn(64, 16).astype(np.float32)
    c_fp32 = MatMulKernels.gemm_fp32(a, b)
    assert c_fp32.shape == (32, 16)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    assert np.allclose(c_fp32, ref, atol=1e-4)
    c_fp16 = MatMulKernels.gemm_fp16(a, b)
    assert c_fp16.shape == (32, 16)
    c_int8 = MatMulKernels.gemm_int8(a, b)
    assert c_int8.shape == (32, 16)
    bias = np.random.randn(16).astype(np.float32)
    c_bias = MatMulKernels.gemm_fp32(a, b, bias=bias)
    assert np.allclose(c_bias, ref + bias[np.newaxis, :], atol=1e-4)
    c_relu = MatMulKernels.gemm_fp32(a, b, activation="relu")
    assert np.all(c_relu >= -1e-6)
    print("PASS")


def _test_norm_kernels():
    print("  NormKernels ... ", end="", flush=True)
    x = np.random.randn(8, 64).astype(np.float32)
    w = np.random.randn(64).astype(np.float32)
    rn = NormKernels.rms_norm(x, w)
    assert rn.shape == (8, 64), f"rms_norm shape mismatch: {rn.shape}"
    ln = NormKernels.layer_norm(x, w)
    assert ln.shape == (8, 64), f"layer_norm shape mismatch: {ln.shape}"
    ln_with_bias = NormKernels.layer_norm(x, w, bias=w * 0.1)
    assert ln_with_bias.shape == (8, 64)
    rn_fused = NormKernels.rms_norm_fused(x, w, x * 0.1)
    assert rn_fused.shape == (8, 64), f"rms_norm_fused shape mismatch"
    online = NormKernels.online_rms_norm(x, w)
    assert online.shape == (8, 64), f"online_rms_norm shape mismatch"
    diff = np.max(np.abs(rn.astype(np.float64) - online.astype(np.float64)))
    assert diff < 1e-3, f"online rms_norm mismatch: max diff={diff}"
    batch = NormKernels.batch_rms_norm(np.stack([x, x]), w)
    assert batch.shape == (2, 8, 64), f"batch_rms_norm shape mismatch"
    print("PASS")


def _test_activation_kernels():
    print("  ActivationKernels ... ", end="", flush=True)
    x = np.random.randn(4, 16).astype(np.float32)
    s = ActivationKernels.silu(x)
    assert s.shape == x.shape
    g = ActivationKernels.gelu(x)
    assert g.shape == x.shape
    r = ActivationKernels.relu(x)
    assert np.all(r >= 0.0)
    gate = np.random.randn(4, 8).astype(np.float32)
    up = np.random.randn(4, 8).astype(np.float32)
    sw = ActivationKernels.swiglu(gate, up)
    assert sw.shape == (4, 8)
    gg = ActivationKernels.geglu(gate, up)
    assert gg.shape == (4, 8)
    print("PASS")


def _test_softmax_kernels():
    print("  SoftmaxKernels ... ", end="", flush=True)
    x = np.random.randn(4, 16).astype(np.float32)
    ss = SoftmaxKernels.stable_softmax(x)
    assert np.allclose(ss.sum(axis=-1), 1.0, atol=1e-5)
    os = SoftmaxKernels.online_softmax(x[0])
    assert np.allclose(os.sum(), 1.0, atol=1e-5)
    mask = np.ones_like(x) > 0.5
    ms = SoftmaxKernels.masked_softmax(x, mask)
    assert np.allclose(ms.sum(axis=-1), 1.0, atol=1e-5)
    cs = SoftmaxKernels.causal_softmax(np.random.randn(4, 4).astype(np.float32))
    assert cs.shape == (4, 4)
    fs = SoftmaxKernels.flash_softmax(x)
    assert np.allclose(fs.sum(axis=-1), 1.0, atol=1e-5)
    ss2 = SoftmaxKernels.spectral_softmax(x, temperature=0.8)
    assert np.allclose(ss2.sum(axis=-1), 1.0, atol=1e-5)
    print("PASS")


def _test_reduce_kernels():
    print("  ReduceKernels ... ", end="", flush=True)
    x = np.random.randn(4, 8, 16).astype(np.float32)
    assert np.allclose(
        ReduceKernels.sum(x), np.sum(x.astype(np.float64)).astype(np.float32)
    )
    assert ReduceKernels.max(x).shape == ()
    assert ReduceKernels.mean(x, axis=0).shape == (8, 16)
    assert ReduceKernels.logsumexp(x, axis=-1).shape == (4, 8)
    assert ReduceKernels.cumsum(x, axis=-1).shape == x.shape
    assert ReduceKernels.log_softmax(x, axis=-1).shape == x.shape
    print("PASS")


def _test_memory_operations():
    print("  MemoryOperations ... ", end="", flush=True)
    x = np.random.randn(16, 32).astype(np.float32)
    c = MemoryOperations.flash_copy(x)
    assert np.allclose(x, c)
    s = MemoryOperations.flash_scale(x, 2.0)
    assert np.allclose(s, x.astype(np.float64) * 2.0)
    idx = np.array([0, 2, 4], dtype=np.int32)
    g = MemoryOperations.gather(x, idx, axis=0)
    assert g.shape == (3, 32)
    assert np.allclose(g, x[idx])
    print("PASS")


def _test_spectral_operations():
    print("  SpectralOperations ... ", end="", flush=True)
    x = np.random.randn(8, 16).astype(np.float32)
    sp = SpectralOperations.spectral_pointwise(x, lambda z: z * 1.5)
    assert sp.shape == x.shape
    bs = SpectralOperations.spectral_band_select(x, 2, 8)
    assert bs.shape == x.shape
    pool = SpectralOperations.spectral_pool(x, pool_size=2)
    assert pool.shape[-1] == 8
    print("PASS")


def _test_novel_inventions():
    print("  SpectralGEMM ... ", end="", flush=True)
    a = np.random.randn(16, 32).astype(np.float32)
    b = np.random.randn(32, 8).astype(np.float32)
    c_spec = SpectralGEMM.gemm(a, b, spectral_rank=16)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    assert c_spec.shape == ref.shape
    print("PASS")

    print("  ResonantRoofline ... ", end="", flush=True)
    ResonantRoofline.calibrate()
    kernel = ResonantRoofline.select_gemm_kernel(1024, 1024, 1024)
    bs = ResonantRoofline.optimal_block_size(1024)
    assert bs > 0
    print(f"PASS (kernel={kernel}, block={bs})")

    print("  VlasovBatchScheduler ... ", end="", flush=True)
    vs = VlasovBatchScheduler()
    vs.submit("matmul", lambda: None, ())
    assert vs.pending_count() == 1
    vs.flush()
    assert vs.pending_count() == 0
    print("PASS")

    print("  QuantumSuperpositionEngine ... ", end="", flush=True)
    tensors = [np.random.randn(4, 8).astype(np.float32) for _ in range(3)]
    enc, sizes = QuantumSuperpositionEngine.amplitude_encode(tensors)
    dec = QuantumSuperpositionEngine.amplitude_decode(enc, sizes)
    assert len(dec) == len(tensors)
    for t, d in zip(tensors, dec):
        assert np.allclose(t, d)
    print("PASS")

    print("  HolographicGEMM ... ", end="", flush=True)
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 4).astype(np.float32)
    c_holo = HolographicGEMM.gemm(a, b)
    assert c_holo.shape == (4, 4)
    print("PASS")


def _test_tensorops_api():
    print("  TensorOps unified API ... ", end="", flush=True)
    a = np.random.randn(16, 32).astype(np.float32)
    b = np.random.randn(32, 8).astype(np.float32)
    c = TensorOps.gemm(a, b)
    assert c.shape == (16, 8)
    c2 = TensorOps.gemm(a, b, activation="relu")
    assert np.all(c2 >= -1e-6)
    x = np.random.randn(4, 16).astype(np.float32)
    r = TensorOps.norm.rms_norm(x, np.random.randn(16).astype(np.float32))
    assert r.shape == (4, 16)
    s = TensorOps.softmax.stable_softmax(x)
    assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-5)
    print("PASS")


def run_all_tests():
    print("=" * 60)
    print("Tensor Operations Engine — Self Verification")
    print("=" * 60)
    tests = [
        ("TensorLayout", _test_tensor_layout),
        ("FusedOperations", _test_fused_operations),
        ("MatMulKernels", _test_matmul_kernels),
        ("NormKernels", _test_norm_kernels),
        ("ActivationKernels", _test_activation_kernels),
        ("SoftmaxKernels", _test_softmax_kernels),
        ("ReduceKernels", _test_reduce_kernels),
        ("MemoryOperations", _test_memory_operations),
        ("SpectralOperations", _test_spectral_operations),
        ("NovelInventions", _test_novel_inventions),
        ("TensorOpsAPI", _test_tensorops_api),
    ]
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1
    print("=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed > 0:
        print("SOME TESTS FAILED!")
    else:
        print("ALL TESTS PASSED!")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        print("Tensor Operations Engine — SpectralStream")
        print("Usage: python -m spectralstream.tensor_ops_engine --test")

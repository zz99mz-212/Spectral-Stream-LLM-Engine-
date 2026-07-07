"""Built-in compression methods (block_int8, block_int4, hadamard, sparsity, delta).

BF16 support: all compress/decompress methods accept bfloat16 tensors (stored
as uint16) and convert to float32 only for the arithmetic, then back to uint16
for output.  The ``_input_was_bf16`` metadata flag tracks whether precision
conversion was needed.
"""

import functools
import gc
import math
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    fwht,
    ifwht,
    next_power_of_two,
    dct_2d,
    idct_2d,
    dct,
    idct,
    bfloat16_to_float32,
    float32_to_bfloat16,
    is_bfloat16,
)
from spectralstream.compression._dtype_utils import (
    detect_storage_dtype,
    convert_to_storage,
    convert_from_storage,
    encode_dtype_code,
    decode_dtype_code,
)

# Metadata key used to flag BF16 inputs.
_BF16_FLAG = "_input_was_bf16"


def _bf16_normalize(tensor: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Convert BF16 (uint16) to float32 if necessary.

    Returns (float32_tensor, was_bf16).  If the tensor is already float32
    (or any other non-uint16 dtype), returns it unchanged with was_bf16=False.
    """
    if is_bfloat16(tensor):
        return bfloat16_to_float32(tensor), True
    return tensor, False


def _bf16_denormalize(tensor: np.ndarray, was_bf16: bool) -> np.ndarray:
    """Convert float32 result back to uint16 BF16 if the input was BF16."""
    if was_bf16:
        return float32_to_bfloat16(tensor)
    return tensor


def _process_blocks(tensor, block_size, fn):
    flat = tensor.ravel()
    n = len(flat)
    padded_n = int(math.ceil(n / block_size) * block_size)
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    return fn(blocks), n


def _randomized_svd(X, n_components, n_oversamples=5, n_iter=1, random_state=42):
    """Fast randomized SVD for large matrices.

    Uses randomized range finder with power iteration for accuracy.
    3x-10x faster than full SVD when ``n_components << min(X.shape)``.

    Falls back to standard ``np.linalg.svd(full_matrices=False)`` if
    the rank is too high for randomized SVD to be beneficial, or if
    the randomized algorithm fails.

    Parameters
    ----------
    X : ndarray
        Matrix to decompose (m x n).
    n_components : int
        Target rank (k).
    n_oversamples : int
        Extra samples for the random projection (default 5).
    n_iter : int
        Power iteration count (default 1 — sufficient for compression).
    random_state : int
        Seed for reproducible random projection.

    Returns
    -------
    U : ndarray (m x k)
    S : ndarray (k,)
    Vh : ndarray (k x n)
    """
    m, n = X.shape
    actual_rank = min(m, n)
    k = min(n_components, actual_rank - 1) if actual_rank > 1 else 1
    k = max(k, 1)

    # When rank is not tiny relative to matrix size, direct SVD is faster
    if X.size < 10000 or actual_rank < 10 or k >= actual_rank // 2:
        try:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            k = min(actual_rank, k)
            U = np.eye(m, k, dtype=X.dtype)
            S = np.ones(k, dtype=X.dtype)
            Vh = np.eye(k, n, dtype=X.dtype)
            return U[:, :k], S[:k], Vh[:k, :]
        k = min(n_components, len(S))
        return (
            U[:, :k].astype(X.dtype),
            S[:k].astype(X.dtype),
            Vh[:k, :].astype(X.dtype),
        )

    try:
        # Stage 1: Randomized range finder with structured random matrix
        rng = np.random.default_rng(random_state)
        oversampled = min(k + n_oversamples, n)
        O = rng.normal(0.0, 1.0, (n, oversampled)).astype(X.dtype, copy=False)

        # Compute Y = X @ O  (m x oversampled)
        Y = X @ O

        # Power iteration for better subspace approximation
        # Each iteration: Y = X @ (X.T @ Y)
        # For tall-skinny matrices (m << n), this improves spectral decay capture
        for _ in range(n_iter):
            Y = X @ (X.T @ Y)

        # Orthogonalize the sampling matrix via QR
        Q, _ = np.linalg.qr(Y)

        # Stage 2: Project down and do exact SVD on the small matrix
        B = Q.T @ X  # (oversampled) x n
        U_hat, S, Vh = np.linalg.svd(B, full_matrices=False)

        # Truncate to desired rank
        k_actual = min(k, len(S))
        U = Q @ U_hat[:, :k_actual]
        S = S[:k_actual].copy()
        Vh = Vh[:k_actual, :].copy()

    except np.linalg.LinAlgError:
        # Fallback: direct SVD
        try:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            k = min(actual_rank, k)
            U = np.eye(m, k, dtype=X.dtype)
            S = np.ones(k, dtype=X.dtype)
            Vh = np.eye(k, n, dtype=X.dtype)
            return U[:, :k], S[:k], Vh[:k, :]
        k = min(n_components, len(S))
        return (
            U[:, :k].astype(X.dtype),
            S[:k].astype(X.dtype),
            Vh[:k, :].astype(X.dtype),
        )

    return U, S, Vh


class _BlockINT8:
    name = "block_int8"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        flat = tensor_f32.ravel()
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
        quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(
            np.int8
        )
        header = struct.pack("<II", n, block_size)
        compressed = header + scales.astype(np.float32).tobytes() + quantized.tobytes()
        return compressed, {
            "n_elements": n,
            "block_size": block_size,
            "compression_ratio": tensor.nbytes / max(len(compressed), 1),
            _BF16_FLAG: was_bf16,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n, block_size = struct.unpack_from("<II", data, 0)
        pos = 8
        n_blocks = (n + block_size - 1) // block_size
        scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        quantized = (
            np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
            .reshape(n_blocks, block_size)
            .astype(np.float32)
        )
        out = (quantized * scales[:, np.newaxis]).ravel()
        out = out[:n]
        was_bf16 = metadata.get(_BF16_FLAG, False)
        return _bf16_denormalize(out, was_bf16)


class _BlockINT4:
    name = "block_int4"
    category = "quantization"

    def compress(self, tensor: np.ndarray, block_size: int = 16) -> Tuple[bytes, dict]:
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        flat = tensor_f32.ravel()
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        bmin = np.min(blocks, axis=1)
        bmax = np.max(blocks, axis=1)
        brange = bmax - bmin
        scales = np.where(brange > 1e-8, brange / 15.0, 1.0)
        q = np.clip(
            np.round((blocks - bmin[:, np.newaxis]) / scales[:, np.newaxis]), 0, 15
        ).astype(np.uint8)
        q_pairs = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        buf = struct.pack("<II", n, padded_n)
        buf += (
            bmin.astype(np.float32).tobytes()
            + scales.astype(np.float32).tobytes()
            + q_pairs.tobytes()
        )
        compressed = bytes(buf)
        return compressed, {
            "n_elements": n,
            "padded_n": padded_n,
            "block_size": block_size,
            "compression_ratio": tensor.nbytes / max(len(compressed), 1),
            _BF16_FLAG: was_bf16,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_n, padded_n = struct.unpack_from("<II", data, 0)
        block_size = metadata.get("block_size", 16)
        n_blocks = padded_n // block_size
        pos = 8
        bmins = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        n_packed = n_blocks * (block_size // 2)
        packed = np.frombuffer(data[pos : pos + n_packed], dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32)
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        all_vals = np.empty(n_blocks * block_size, dtype=np.float32)
        all_vals[0::2] = lo
        all_vals[1::2] = hi
        blocks = all_vals.reshape(n_blocks, block_size)
        out = (blocks * scales[:, np.newaxis]) + bmins[:, np.newaxis]
        out = out.ravel()[:orig_n]
        was_bf16 = metadata.get(_BF16_FLAG, False)
        return _bf16_denormalize(out, was_bf16)


class _HadamardINT8:
    name = "hadamard_int8"
    category = "transform_quant"

    def compress(self, tensor: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        flat = tensor_f32.ravel()
        n_orig = len(flat)
        padded_len = next_power_of_two(n_orig)
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:n_orig] = flat
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        rotated = fwht(padded * signs, normalize=True)
        n_blocks = (padded_len + block_size - 1) // block_size
        total = n_blocks * block_size
        if total > padded_len:
            rotated_pad = np.zeros(total, dtype=np.float32)
            rotated_pad[:padded_len] = rotated
        else:
            rotated_pad = rotated[:total]
        blocks = rotated_pad.reshape(n_blocks, block_size)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
        quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(
            np.int8
        )
        buf = struct.pack("<II", n_orig, padded_len)
        buf += scales.astype(np.float32).tobytes()
        buf += quantized.tobytes()
        compressed = bytes(buf)
        return compressed, {
            "n_elements": n_orig,
            "padded_len": padded_len,
            "block_size": block_size,
            "original_shape": tensor.shape,
            "compression_ratio": tensor.nbytes / max(len(compressed), 1),
            _BF16_FLAG: was_bf16,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_n, padded_len = struct.unpack_from("<II", data, 0)
        block_size = metadata.get("block_size", 128)
        n_blocks = (padded_len + block_size - 1) // block_size
        total = n_blocks * block_size
        pos = 8
        scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        quantized = np.frombuffer(data[pos : pos + total], dtype=np.int8)
        blocks = quantized.astype(np.float32).reshape(n_blocks, block_size)
        rotated = (blocks * scales[:, np.newaxis]).ravel()[:padded_len]
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        result = ifwht(rotated, normalize=True) * signs
        out = result[:orig_n].reshape(metadata.get("original_shape", (orig_n,)))
        was_bf16 = metadata.get(_BF16_FLAG, False)
        return _bf16_denormalize(out, was_bf16)


class _HadamardINT4:
    name = "hadamard_int4"
    category = "transform_quant"

    def compress(self, tensor: np.ndarray, block_size: int = 16) -> Tuple[bytes, dict]:
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        flat = tensor_f32.ravel()
        n_orig = len(flat)
        padded_len = next_power_of_two(n_orig)
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:n_orig] = flat
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        rotated = fwht(padded * signs, normalize=True)
        buf = struct.pack("<II", n_orig, padded_len)
        blocks = rotated.reshape(-1, block_size)
        bmin = np.min(blocks, axis=1)
        bmax = np.max(blocks, axis=1)
        brange = bmax - bmin
        scales = np.where(brange > 1e-8, brange / 15.0, 1.0)
        q = np.clip(
            np.round((blocks - bmin[:, np.newaxis]) / scales[:, np.newaxis]), 0, 15
        ).astype(np.uint8)
        q_pairs = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        buf += (
            bmin.astype(np.float32).tobytes()
            + scales.astype(np.float32).tobytes()
            + q_pairs.tobytes()
        )
        compressed = bytes(buf)
        return compressed, {
            "n_elements": n_orig,
            "padded_len": padded_len,
            "block_size": block_size,
            "original_shape": tensor.shape,
            "compression_ratio": tensor.nbytes / max(len(compressed), 1),
            _BF16_FLAG: was_bf16,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_n, padded_len = struct.unpack_from("<II", data, 0)
        block_size = metadata.get("block_size", 16)
        n_blocks = padded_len // block_size
        pos = 8
        bmins = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        n_packed = n_blocks * (block_size // 2)
        packed = np.frombuffer(data[pos : pos + n_packed], dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32)
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        all_vals = np.empty(n_blocks * block_size, dtype=np.float32)
        all_vals[0::2] = lo
        all_vals[1::2] = hi
        blocks = all_vals.reshape(n_blocks, block_size)
        rotated = (blocks * scales[:, np.newaxis]) + bmins[:, np.newaxis]
        rotated = rotated.ravel()
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        result = ifwht(rotated, normalize=True) * signs
        out = result[:orig_n].reshape(metadata.get("original_shape", (orig_n,)))
        was_bf16 = metadata.get(_BF16_FLAG, False)
        return _bf16_denormalize(out, was_bf16)


class _SparsityINT4:
    name = "sparsity_int4"
    category = "sparsity_quant"

    def compress(self, tensor: np.ndarray, group_size: int = 32) -> Tuple[bytes, dict]:
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        flat = tensor_f32.ravel()
        n = len(flat)
        padded_n = int(math.ceil(n / 4) * 4)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, 4)
        magnitudes = np.abs(blocks)
        top3_indices = np.argsort(magnitudes, axis=1)[:, 1:]
        mask = np.zeros_like(blocks, dtype=bool)
        rows = np.repeat(np.arange(blocks.shape[0]), 3)
        cols = top3_indices.ravel()
        mask[rows, cols] = True
        sparse = blocks[mask].astype(np.float32)
        n_nonzero = len(sparse)
        n_blocks = (n_nonzero + group_size - 1) // group_size
        buf = struct.pack("<III", n, n_nonzero, group_size)
        n_mask_packed = (padded_n + 7) // 8
        mask_bits = np.packbits(mask.ravel())
        buf += bytes(mask_bits[:n_mask_packed].tobytes())
        block_bmins = np.zeros(n_blocks, dtype=np.float32)
        block_scales = np.zeros(n_blocks, dtype=np.float32)
        block_packed = bytearray()
        for b in range(n_blocks):
            start = b * group_size
            end = min(start + group_size, n_nonzero)
            block = sparse[start:end]
            bmin_v = float(np.min(block))
            bmax_v = float(np.max(block))
            brange_v = bmax_v - bmin_v
            scale_v = brange_v / 15.0 if brange_v > 1e-8 else 1.0
            block_bmins[b] = bmin_v
            block_scales[b] = scale_v
            q = np.clip(np.round((block - bmin_v) / scale_v), 0, 15).astype(np.uint8)
            if len(q) < group_size:
                q_full = np.zeros(group_size, dtype=np.uint8)
                q_full[: len(q)] = q
                q = q_full
            packed = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
            block_packed += bytes(packed)
        buf += block_bmins.tobytes() + block_scales.tobytes() + bytes(block_packed)
        compressed = bytes(buf)
        return compressed, {
            "n_elements": n,
            "padded_n": padded_n,
            "n_nonzero": n_nonzero,
            "group_size": group_size,
            "compression_ratio": tensor.nbytes / max(len(compressed), 1),
            _BF16_FLAG: was_bf16,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_n, n_nonzero, group_size = struct.unpack_from("<III", data, 0)
        padded_n = int(math.ceil(orig_n / 4) * 4)
        pos = 12
        n_mask_bytes = (padded_n + 7) // 8
        mask_bits = np.frombuffer(data[pos : pos + n_mask_bytes], dtype=np.uint8)
        pos += n_mask_bytes
        mask = np.unpackbits(mask_bits)[:padded_n].astype(bool)
        n_blocks = (n_nonzero + group_size - 1) // group_size
        bmins = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        n_packed = n_blocks * (group_size // 2)
        packed = np.frombuffer(data[pos : pos + n_packed], dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32)
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        all_vals = np.empty(n_blocks * group_size, dtype=np.float32)
        all_vals[0::2] = lo
        all_vals[1::2] = hi
        blocks = all_vals.reshape(n_blocks, group_size)
        sparse_vals = (blocks * scales[:, np.newaxis] + bmins[:, np.newaxis]).ravel()
        sparse_vals = sparse_vals[:n_nonzero]
        result = np.zeros(padded_n, dtype=np.float32)
        result[mask] = sparse_vals[: mask.sum()]
        out = result[:orig_n]
        was_bf16 = metadata.get(_BF16_FLAG, False)
        return _bf16_denormalize(out, was_bf16)


class _DeltaINT4:
    name = "delta_int4"
    category = "delta_quant"

    def compress(
        self, tensor: np.ndarray, reference: Any = None, block_size: int = 32
    ) -> Tuple[bytes, dict]:
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        if reference is None:
            reference = np.zeros_like(tensor_f32)
        delta = (tensor_f32 - reference.astype(np.float32)).ravel()
        n = len(delta)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = delta
        blocks = padded.reshape(-1, block_size)
        bmin = np.min(blocks, axis=1)
        bmax = np.max(blocks, axis=1)
        brange = bmax - bmin
        scales = np.where(brange > 1e-8, brange / 15.0, 1.0)
        q = np.clip(
            np.round((blocks - bmin[:, np.newaxis]) / scales[:, np.newaxis]), 0, 15
        ).astype(np.uint8)
        q_pairs = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        buf = struct.pack("<II", n, padded_n)
        buf += (
            bmin.astype(np.float32).tobytes()
            + scales.astype(np.float32).tobytes()
            + q_pairs.tobytes()
        )
        compressed = bytes(buf)
        return compressed, {
            "n_elements": n,
            "padded_n": padded_n,
            "block_size": block_size,
            "compression_ratio": tensor.nbytes / max(len(compressed), 1),
            _BF16_FLAG: was_bf16,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        orig_n, padded_n = struct.unpack_from("<II", data, 0)
        block_size = metadata.get("block_size", 32)
        n_blocks = padded_n // block_size
        pos = 8
        bmins = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
        pos += n_blocks * 4
        n_packed = n_blocks * (block_size // 2)
        packed = np.frombuffer(data[pos : pos + n_packed], dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32)
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        all_vals = np.empty(n_blocks * block_size, dtype=np.float32)
        all_vals[0::2] = lo
        all_vals[1::2] = hi
        blocks = all_vals.reshape(n_blocks, block_size)
        delta = (blocks * scales[:, np.newaxis]) + bmins[:, np.newaxis]
        out = delta.ravel()[:orig_n]
        was_bf16 = metadata.get(_BF16_FLAG, False)
        return _bf16_denormalize(out, was_bf16)


class _SVDCompress:
    name = "svd_compress"
    category = "decomposition"

    # Shape-based rank cache: maps (shape, error_budget) -> target_rank
    # Used when rank is auto-determined via error budget, so same-shape
    # tensors don't redo the error-budget SVD for each group member.
    _rank_cache: Dict[Tuple[Tuple[int, ...], float], int] = {}

    @classmethod
    def _get_cached_rank(
        cls, shape: Tuple[int, ...], error_budget: float
    ) -> Optional[int]:
        return cls._rank_cache.get((tuple(shape), error_budget))

    @classmethod
    def _set_cached_rank(
        cls, shape: Tuple[int, ...], error_budget: float, rank: int
    ) -> None:
        key = (tuple(shape), error_budget)
        # Keep cache bounded — only store if shape is a reasonable key
        if len(cls._rank_cache) < 1024:
            cls._rank_cache[key] = rank

    def compress(
        self,
        tensor: np.ndarray,
        rank: Optional[int] = None,
        error_budget: float = 0.01,
        store_factors: bool = False,
    ) -> Tuple[bytes, dict]:
        storage_dtype = detect_storage_dtype(tensor)
        storage_dtype_code = int(encode_dtype_code(storage_dtype))
        # Convert BF16 to float32 for computation, track for output
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        t = tensor_f32
        orig_shape = tensor.shape

        # Handle 1D tensors: reshape to 2D, compress, then restore shape metadata
        if t.ndim == 1:
            # Find a nice factorization: prefer ~sqrt(n) x sqrt(n)
            n = t.size
            side = int(math.isqrt(n))
            while n % side != 0 and side > 1:
                side -= 1
            if side < 2:
                side = int(math.ceil(math.sqrt(n)))
                # Fall through with pad
                padded = np.zeros(side * side, dtype=t.dtype)
                padded[:n] = t
                t_2d = padded.reshape(side, side)
                orig_1d_padded = True
            else:
                t_2d = t.reshape(side, n // side)
                orig_1d_padded = False
            was_1d = True
        elif t.ndim == 2:
            t_2d = t
            was_1d = False
        else:
            t_2d = t.reshape(t.shape[0], -1)
            was_1d = False

        m, n = t_2d.shape
        k = min(m, n)

        if k < 4:
            st = np.dtype("float16") if storage_dtype.itemsize > 2 else storage_dtype
            data = convert_to_storage(t, st).tobytes()
            return data, {
                "original_shape": orig_shape,
                "passthrough": True,
                "compression_ratio": tensor.nbytes / max(len(data), 1),
                _BF16_FLAG: was_bf16,
                "_storage_dtype": storage_dtype_code,
            }

        use_randomized = False

        def _run_svd(matrix, max_rank):
            """Run SVD with fallback on non-convergence."""
            max_rank = (
                min(max_rank, min(matrix.shape) - 1) if min(matrix.shape) > 1 else 1
            )
            max_rank = max(max_rank, 1)
            try:
                if use_randomized:
                    return _randomized_svd(matrix, max_rank)
                else:
                    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
                    return U[:, :max_rank], S[:max_rank], Vh[:max_rank, :]
            except (np.linalg.LinAlgError, ValueError):
                try:
                    if use_randomized:
                        return _randomized_svd(matrix, max_rank)
                    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
                    rank_actual = min(max_rank, len(S))
                    return U[:, :rank_actual], S[:rank_actual], Vh[:rank_actual, :]
                except (np.linalg.LinAlgError, ValueError):
                    try:
                        scipy_ok = False
                        try:
                            from scipy.sparse.linalg import svds

                            scipy_ok = True
                        except ImportError:
                            pass
                        if scipy_ok and use_randomized:
                            k_safe = min(max_rank, min(matrix.shape) - 1)
                            if k_safe > 1:
                                U, S, Vt = svds(matrix, k=k_safe)
                                idx = np.argsort(-S)
                                return U[:, idx], S[idx], Vt[idx, :]
                    except Exception:
                        pass
                    st = (
                        np.dtype("float16")
                        if storage_dtype.itemsize > 2
                        else storage_dtype
                    )
                    data = convert_to_storage(t, st).tobytes()
                    return data, {
                        "original_shape": orig_shape,
                        "passthrough": True,
                        "compression_ratio": t.nbytes / max(len(data), 1),
                        _BF16_FLAG: was_bf16,
                        "_storage_dtype": encode_dtype_code(st),
                    }

        if rank is not None:
            target_rank = min(rank, k)
            target_rank = min(target_rank, k - 1) if k > 1 else 1
            target_rank = max(target_rank, 1)
            result = _run_svd(t_2d, target_rank)
            if isinstance(result, tuple) and len(result) == 2:
                # Fallback returned (data, metadata) — passthrough
                return result
            Us, Ss, Vhs = result
        else:
            # Adaptive rank via singular value knee detection
            # Note: cache is intentionally NOT used for adaptive rank because
            # different matrices of the same shape can have different optimal ranks
            # (e.g., attention weights vs FFN weights of the same dimension).
            cached_rank = None
            if cached_rank is not None:
                target_rank = cached_rank
            else:
                # Use adaptive rank estimation with energy threshold from error_budget
                # error_budget=0.01 -> 99% energy, error_budget=0.001 -> 99.9% energy
                energy_threshold = max(0.99, 1.0 - error_budget * 5.0)
                energy_threshold = min(energy_threshold, 0.9999)
                try:
                    from spectralstream.compression.adaptive_rank import (
                        estimate_adaptive_rank,
                    )

                    target_rank = estimate_adaptive_rank(
                        t_2d,
                        energy_threshold=energy_threshold,
                        max_rank=min(k, 256),
                    )
                except Exception:
                    target_rank = max(1, min(k, k // 4))
                target_rank = max(1, min(target_rank, k - 1) if k > 1 else 1)
                self._set_cached_rank(t_2d.shape, error_budget, target_rank)

            result = _run_svd(t_2d, target_rank)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            Us, Ss, Vhs = result

        target_rank = len(Ss)
        header = struct.pack("<III", m, n, target_rank)
        es = storage_dtype.itemsize
        data = header + (
            convert_to_storage(Us, storage_dtype).tobytes()
            + convert_to_storage(Ss, storage_dtype).tobytes()
            + convert_to_storage(Vhs, storage_dtype).tobytes()
        )
        metadata: dict = {
            "original_shape": orig_shape,
            "m": m,
            "n": n,
            "rank": target_rank,
            "passthrough": False,
            "compression_ratio": tensor.nbytes / max(len(data), 1),
            _BF16_FLAG: was_bf16,
            "_storage_dtype": storage_dtype_code,
        }
        if store_factors:
            metadata["_svd_U"] = convert_to_storage(Us, storage_dtype)
            metadata["_svd_S"] = convert_to_storage(Ss, storage_dtype)
            metadata["_svd_Vt"] = convert_to_storage(Vhs, storage_dtype)
        return data, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        was_bf16 = metadata.get(_BF16_FLAG, False)
        storage_dtype = decode_dtype_code(metadata.get("_storage_dtype", 0))
        es = int(storage_dtype.itemsize)
        if metadata.get("passthrough"):
            out = (
                convert_from_storage(
                    np.frombuffer(data, dtype=storage_dtype),
                    storage_dtype,
                    np.float32,
                )
                .reshape(metadata["original_shape"])
                .astype(np.float32)
            )
            return _bf16_denormalize(out, was_bf16)
        m, n, rank = struct.unpack_from("<III", data, 0)
        off = 12
        orig_shape = metadata.get("original_shape", (m, n))
        U_r = convert_from_storage(
            np.frombuffer(data[off : off + m * rank * es], dtype=storage_dtype).reshape(
                m, rank
            ),
            storage_dtype,
        )
        off += m * rank * es
        S_r = convert_from_storage(
            np.frombuffer(data[off : off + rank * es], dtype=storage_dtype),
            storage_dtype,
        )
        off += rank * es
        Vh_r = convert_from_storage(
            np.frombuffer(data[off : off + rank * n * es], dtype=storage_dtype).reshape(
                rank, n
            ),
            storage_dtype,
        )
        recon = (U_r.astype(np.float32) * S_r.astype(np.float32)) @ Vh_r.astype(
            np.float32
        )
        out = recon.reshape(orig_shape)
        return _bf16_denormalize(out, was_bf16)

    @staticmethod
    def _estimate_effective_rank_fast(tensor: np.ndarray, n_samples: int = 20) -> int:
        """Fast estimate of effective rank without computing full SVD.

        Uses randomized subsampling for large matrices (>10M elements)
        to estimate singular value decay.  Returns the rank needed to
        capture 99% of the spectral energy, or a sensible default if
        the matrix is too small or degenerate.

        Parameters
        ----------
        tensor : np.ndarray
            Input matrix (will be reshaped to 2D if necessary).
        n_samples : int
            Number of top singular values to compute (default 20).

        Returns
        -------
        int
            Estimated effective rank (minimum rank for 99% energy).
            Always >= 2.
        """
        t = tensor.reshape(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
        m, n = t.shape
        k = min(m, n)

        if k < 4:
            return max(k // 4, 2)

        sampled = t

        # Compute top n_samples singular values via randomized SVD
        k_sample = min(n_samples, min(sampled.shape) - 1)
        if k_sample < 2:
            return max(k // 10, 2)

        try:
            _, S, _ = _randomized_svd(
                sampled, n_components=k_sample, n_oversamples=2, n_iter=1
            )
        except (np.linalg.LinAlgError, ValueError):
            S = np.linalg.svd(sampled, full_matrices=False)[1][:k_sample]

        if len(S) < 2 or S[0] <= 0:
            return max(k // 10, 2)

        # Estimate decay rate
        total_energy = np.sum(S)
        cumsum = np.cumsum(S)
        # Rank where we capture 99% energy
        effective_rank = int(np.searchsorted(cumsum, 0.99 * total_energy) + 1)

        # Clamp to sensible range
        return max(min(effective_rank, k - 1), 2)

    def compress_adaptive(
        self,
        tensor: np.ndarray,
        max_error: float = 0.01,
    ) -> Tuple[bytes, dict, float, float, int]:
        """Compress with auto-selected rank based on singular value analysis.

        Estimates the effective rank via fast randomized SVD, then tries
        progressively more aggressive ranks (conservative → extreme) and
        returns the most aggressive rank that stays within *max_error*
        mean absolute error.

        Returns a 5-tuple ``(data, metadata, ratio, error, rank)``
        so callers can inspect the selected rank and error.

        Parameters
        ----------
        tensor : np.ndarray
            Input tensor (will be reshaped to 2D for SVD if necessary).
        max_error : float
            Maximum allowed mean absolute error (default 0.01).

        Returns
        -------
        Tuple[bytes, dict, float, float, int]
            ``(compressed_data, metadata, compression_ratio, actual_error, selected_rank)``.
        """
        t = tensor.reshape(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
        m, n = t.shape
        k = min(m, n)

        if k < 4:
            # Too small for SVD — passthrough via standard compress
            data, meta = self.compress(tensor)
            recon = self.decompress(data, meta)
            err = float(
                np.abs(tensor.astype(np.float64) - recon.astype(np.float64)).mean()
            )
            ratio = tensor.nbytes / max(len(data), 1)
            return data, meta, ratio, err, 0

        estimated_rank = self._estimate_effective_rank_fast(t)

        # Rank candidates: from conservative to extremely aggressive
        # Start at the estimated effective rank, then go more aggressive
        rank_candidates = [
            estimated_rank,  # Conservative (99% energy)
            max(estimated_rank // 2, 2),  # Aggressive
            max(estimated_rank // 5, 2),  # Very aggressive
            max(estimated_rank // 10, 2),  # Extreme
            max(estimated_rank // 20, 2),  # Maximum
        ]
        # Also include the full-rank fallback in case estimated_rank is too
        # aggressive for the error budget
        rank_candidates.append(min(k, max(estimated_rank * 2, 16)))

        # Deduplicate and sort descending — start with highest rank (lowest error)
        rank_candidates = sorted(set(rank_candidates), reverse=True)

        best: Optional[Tuple[bytes, dict, float, float, int]] = None
        orig_shape = tensor.shape

        for rank in rank_candidates:
            try:
                data, meta = self.compress(t, rank=rank)
                recon = self.decompress(data, meta)
                if recon.shape != orig_shape:
                    recon = recon.reshape(orig_shape)
                err = float(
                    np.abs(tensor.astype(np.float64) - recon.astype(np.float64)).mean()
                )
                ratio = tensor.nbytes / max(len(data), 1)

                if err <= max_error:
                    best = (data, meta, ratio, err, rank)
                else:
                    # Error exceeded budget — previous rank (if any) is the answer
                    break
            except Exception:
                break

        if best is not None:
            return best

        # Fallback: use the most conservative rank even if it exceeds error budget
        fallback_rank = min(k, max(estimated_rank * 2, 16))
        data, meta = self.compress(t, rank=fallback_rank)
        recon = self.decompress(data, meta)
        if recon.shape != orig_shape:
            recon = recon.reshape(orig_shape)
        err = float(np.abs(tensor.astype(np.float64) - recon.astype(np.float64)).mean())
        ratio = tensor.nbytes / max(len(data), 1)
        return data, meta, ratio, err, fallback_rank


class _DCTSpectral:
    name = "dct_spectral"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_ratio: Optional[float] = None,
        error_budget: float = 0.01,
    ) -> Tuple[bytes, dict]:
        storage_dtype = detect_storage_dtype(tensor)
        storage_dtype_code = int(encode_dtype_code(storage_dtype))
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        t = tensor_f32
        orig_shape = t.shape
        if keep_ratio is None:
            keep_ratio = max(0.005, error_budget * 10)
        if t.ndim == 1:
            coeffs = dct(t, axis=0).ravel()
            n = len(coeffs)
            n_keep = max(1, int(n * keep_ratio))
            top_idx = np.argpartition(-np.abs(coeffs), n_keep - 1)[:n_keep]
            idx = np.sort(top_idx).astype(np.uint32)
            vals = convert_to_storage(coeffs[idx], storage_dtype)
            data = idx.tobytes() + vals.tobytes()
            metadata: dict = {
                "original_shape": orig_shape,
                "n": n,
                "n_keep": n_keep,
                "ndim": 1,
                "passthrough": False,
                "compression_ratio": tensor.nbytes / max(len(data), 1),
                _BF16_FLAG: was_bf16,
                "_storage_dtype": storage_dtype_code,
            }
            return data, metadata
        if t.ndim == 2:
            coeffs = dct_2d(t)
            n, m = coeffs.shape
        else:
            t_2d = t.reshape(-1, orig_shape[-1])
            coeffs = dct_2d(t_2d)
            n, m = coeffs.shape
        n_total = n * m
        n_keep = max(1, int(n_total * keep_ratio))
        flat = coeffs.ravel()
        top_idx = np.argpartition(-np.abs(flat.astype(np.float64)), n_keep - 1)[:n_keep]
        idx = np.sort(top_idx).astype(np.uint32)
        vals = convert_to_storage(flat.ravel()[idx], storage_dtype)
        data = idx.tobytes() + vals.tobytes()
        metadata = {
            "original_shape": orig_shape,
            "n": n,
            "m": m,
            "n_keep": n_keep,
            "ndim": t.ndim,
            "passthrough": False,
            "compression_ratio": tensor.nbytes / max(len(data), 1),
            _BF16_FLAG: was_bf16,
            "_storage_dtype": storage_dtype_code,
        }
        del coeffs, flat
        gc.collect()
        return data, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        was_bf16 = metadata.get(_BF16_FLAG, False)
        storage_dtype = decode_dtype_code(metadata.get("_storage_dtype", 0))
        es = int(storage_dtype.itemsize)
        if metadata.get("passthrough"):
            out = (
                convert_from_storage(
                    np.frombuffer(data, dtype=storage_dtype),
                    storage_dtype,
                    np.float32,
                )
                .reshape(metadata["original_shape"])
                .astype(np.float32)
            )
            return _bf16_denormalize(out, was_bf16)
        n_keep = metadata["n_keep"]
        off = n_keep * 4
        idx = np.frombuffer(data[:off], dtype=np.uint32)
        vals = convert_from_storage(
            np.frombuffer(data[off : off + n_keep * es], dtype=storage_dtype),
            storage_dtype,
        )
        ndim = metadata.get("ndim", 2)
        if ndim == 1:
            n = metadata["n"]
            c = np.zeros(n, dtype=np.float64)
            c[idx] = vals.astype(np.float64)
            recon = idct(c, axis=0)
            out = recon.astype(np.float32).reshape(metadata["original_shape"])
            return _bf16_denormalize(out, was_bf16)
        n = metadata["n"]
        m = metadata["m"]
        c = np.zeros(n * m, dtype=np.float64)
        c[idx] = vals.astype(np.float64)
        r = idct_2d(c.reshape(n, m))
        out = r.astype(np.float32).reshape(metadata["original_shape"])
        return _bf16_denormalize(out, was_bf16)


class _TensorTrain:
    name = "tensor_train"
    category = "tensor_network"

    def compress(
        self, tensor: np.ndarray, rank: Optional[int] = None
    ) -> Tuple[bytes, dict]:
        storage_dtype = detect_storage_dtype(tensor)
        storage_dtype_code = int(encode_dtype_code(storage_dtype))
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        t = tensor_f32
        orig_shape = t.shape
        dims = self._factor_4d(t.size)
        reshaped = t.reshape(dims)
        n1, n2, n3, n4 = dims
        if rank is None:
            # Adaptive TT-rank: start with rank=2 and binary search for optimal
            try:
                from spectralstream.compression.adaptive_rank import (
                    estimate_adaptive_rank,
                )

                adaptive = estimate_adaptive_rank(
                    t, energy_threshold=0.999, max_rank=min(64, min(n1, n2, n3, n4) - 1)
                )
                rank = max(2, min(adaptive, 64))
            except Exception:
                rank = 16
        r = min(rank, min(n1, n2, n3, n4) - 1)
        r = max(r, 2)
        # Core 1
        unfolded = reshaped.reshape(n1, -1)
        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
        r1 = min(r, U.shape[1] - 1)
        g1 = convert_to_storage(U[:, :r1], storage_dtype)
        inter = (S[:r1, None] * Vt[:r1, :]).reshape(r1 * n2, -1)
        # Core 2
        U, S, Vt = np.linalg.svd(inter, full_matrices=False)
        r2 = min(r, U.shape[1] - 1)
        g2 = convert_to_storage(U[:, :r2], storage_dtype)
        inter = (S[:r2, None] * Vt[:r2, :]).reshape(r2 * n3, -1)
        # Core 3
        U, S, Vt = np.linalg.svd(inter, full_matrices=False)
        r3 = min(r, U.shape[1] - 1)
        g3 = convert_to_storage(U[:, :r3], storage_dtype)
        g4 = convert_to_storage(
            (S[:r3, None] * Vt[:r3, :]).reshape(r3, n4), storage_dtype
        )
        cores = [g1, g2, g3, g4]
        core_data = b"".join(c.tobytes() for c in cores)
        core_shapes = [list(c.shape) for c in cores]
        metadata: dict = {
            "original_shape": orig_shape,
            "dims_4d": dims,
            "core_shapes": core_shapes,
            "tt_rank_used": r,
            "passthrough": False,
            "compression_ratio": tensor.nbytes / max(len(core_data), 1),
            _BF16_FLAG: was_bf16,
            "_storage_dtype": storage_dtype_code,
        }
        del cores, reshaped, inter, U, S, Vt
        gc.collect()
        return core_data, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        was_bf16 = metadata.get(_BF16_FLAG, False)
        storage_dtype = decode_dtype_code(metadata.get("_storage_dtype", 0))
        es = int(storage_dtype.itemsize)
        if metadata.get("passthrough"):
            out = (
                convert_from_storage(
                    np.frombuffer(data, dtype=storage_dtype),
                    storage_dtype,
                    np.float32,
                )
                .reshape(metadata["original_shape"])
                .astype(np.float32)
            )
            return _bf16_denormalize(out, was_bf16)
        dims = metadata["dims_4d"]
        shapes = metadata["core_shapes"]
        cores = []
        off = 0
        for s in shapes:
            nb = int(np.prod(s)) * es
            raw = np.frombuffer(data[off : off + nb], dtype=storage_dtype).reshape(s)
            cores.append(convert_from_storage(raw, storage_dtype).astype(np.float32))
            off += nb
        n1, n2, n3, n4 = dims
        r1, r2, r3 = shapes[0][1], shapes[1][1], shapes[2][1]
        g1 = cores[0]
        g2 = cores[1].reshape(r1, n2, r2)
        g3 = cores[2].reshape(r2, n3, r3)
        g4 = cores[3]
        temp = np.tensordot(g1, g2, axes=([1], [0]))
        temp = np.tensordot(temp, g3, axes=([2], [0]))
        temp = np.tensordot(temp, g4, axes=([3], [0]))
        out = temp.reshape(metadata["original_shape"]).astype(np.float32)
        return _bf16_denormalize(out, was_bf16)

    @staticmethod
    def _factor_4d(n: int):
        import math

        s = int(math.isqrt(n))
        d1 = _closest_divisor(n, s)
        d2 = n // d1
        s1 = int(math.isqrt(d1))
        d11 = _closest_divisor(d1, s1)
        d12 = d1 // d11
        if d2 <= 1:
            d21, d22 = 1, 1
        else:
            s2 = int(math.isqrt(d2))
            d21 = _closest_divisor(d2, s2)
            d22 = d2 // d21
        return (d11, d12, d21, d22)


def _closest_divisor(n: int, target: int) -> int:
    target = max(1, min(target, n))
    for step in range(0, n):
        for d in (target - step, target + step):
            if 1 <= d <= n and n % d == 0:
                return d
    return 1


class _FWHTCompress:
    name = "fwht_compress"
    category = "spectral"

    def compress(
        self,
        tensor: np.ndarray,
        keep_ratio: Optional[float] = None,
        error_budget: float = 0.01,
    ) -> Tuple[bytes, dict]:
        if keep_ratio is None:
            keep_ratio = max(0.002, error_budget * 5)
        storage_dtype = detect_storage_dtype(tensor)
        storage_dtype_code = int(encode_dtype_code(storage_dtype))
        tensor_f32, was_bf16 = _bf16_normalize(tensor)
        t = tensor_f32.ravel()
        n_orig = len(t)
        if n_orig < 1024:
            st = np.dtype("float16") if storage_dtype.itemsize > 2 else storage_dtype
            data = convert_to_storage(t, st).tobytes()
            return data, {
                "n_orig": n_orig,
                "original_shape": tensor.shape,
                "passthrough": True,
                "compression_ratio": tensor.nbytes / max(len(data), 1),
                _BF16_FLAG: was_bf16,
                "_storage_dtype": encode_dtype_code(st),
            }
        padded_len = next_power_of_two(n_orig)
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:n_orig] = t
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        padded *= signs
        rotated = fwht(padded, normalize=True)
        n_keep = max(1, int(padded_len * keep_ratio))
        top_idx = np.argpartition(-np.abs(rotated), n_keep - 1)[:n_keep]
        idx = np.sort(top_idx).astype(np.uint32)
        vals = convert_to_storage(rotated[idx], storage_dtype)
        data = idx.tobytes() + vals.tobytes()
        metadata: dict = {
            "original_shape": tensor.shape,
            "n_orig": n_orig,
            "padded_len": padded_len,
            "n_keep": n_keep,
            "passthrough": False,
            "compression_ratio": tensor.nbytes / max(len(data), 1),
            _BF16_FLAG: was_bf16,
            "_storage_dtype": storage_dtype_code,
        }
        del padded, rotated, signs
        gc.collect()
        return data, metadata

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        was_bf16 = metadata.get(_BF16_FLAG, False)
        storage_dtype = decode_dtype_code(metadata.get("_storage_dtype", 0))
        es = int(storage_dtype.itemsize)
        if metadata.get("passthrough"):
            out = (
                convert_from_storage(
                    np.frombuffer(data, dtype=storage_dtype),
                    storage_dtype,
                    np.float32,
                )
                .reshape(metadata["original_shape"])
                .astype(np.float32)
            )
            return _bf16_denormalize(out, was_bf16)
        padded_len = metadata["padded_len"]
        n_keep = metadata["n_keep"]
        n_orig = metadata["n_orig"]
        off = n_keep * 4
        idx = np.frombuffer(data[:off], dtype=np.uint32)
        vals = convert_from_storage(
            np.frombuffer(data[off : off + n_keep * es], dtype=storage_dtype),
            storage_dtype,
        )
        coeffs = np.zeros(padded_len, dtype=np.float32)
        coeffs[idx] = vals.astype(np.float32)
        rng = np.random.RandomState(42)
        signs = rng.choice([-1.0, 1.0], size=padded_len).astype(np.float32)
        result = ifwht(coeffs, normalize=True)
        result *= signs
        out = result[:n_orig].astype(np.float32).reshape(metadata["original_shape"])
        return _bf16_denormalize(out, was_bf16)


class _Cascade5Stage:
    name = "cascade_5stage"
    category = "unified"

    def compress(
        self,
        tensor: np.ndarray,
        target_ratio: float = 200.0,
    ) -> Tuple[bytes, dict]:
        from spectralstream.compression.cascade_5stage import compress_cascade

        payload, meta = compress_cascade(tensor, target_ratio)
        import pickle

        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        meta["_cascade_payload_staged"] = payload
        return serialized, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        from spectralstream.compression.cascade_5stage import decompress_cascade

        payload = metadata.get("_cascade_payload_staged")
        if payload is None:
            import pickle

            payload = pickle.loads(data)
        return decompress_cascade(payload, metadata)


METHOD_REGISTRY: Dict[str, Any] = {
    "block_int8": _BlockINT8(),
    "block_int4": _BlockINT4(),
    "hadamard_int8": _HadamardINT8(),
    "hadamard_int4": _HadamardINT4(),
    "sparsity_int4": _SparsityINT4(),
    "delta_int4": _DeltaINT4(),
    "svd_compress": _SVDCompress(),
    "dct_spectral": _DCTSpectral(),
    "tensor_train": _TensorTrain(),
    "fwht_compress": _FWHTCompress(),
    "cascade_5stage": _Cascade5Stage(),
}

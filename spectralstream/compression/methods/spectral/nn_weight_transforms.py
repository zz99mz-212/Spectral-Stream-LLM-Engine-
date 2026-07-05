"""Neural network weight transform compression — 20 transform-domain techniques.

Each applies an orthogonal/basis transform to decorrelate weights,
then quantizes the transform coefficients (INT8/INT4).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    fwht,
    idct,
    next_power_of_two,
)

logger = logging.getLogger(__name__)


class NNWeightTransform(ABC):
    """Base class for all neural network weight transform techniques."""

    METHOD_NAME: str = "base_nn_transform"

    @abstractmethod
    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...

    @abstractmethod
    def decompress(
        self, data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> np.ndarray: ...

    def _compressed_bytes(self, data: Dict[str, Any]) -> int:
        total = 0
        for v in data.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
            elif isinstance(v, (int, np.int32, np.int64)):
                total += 4
            elif isinstance(v, (float, np.float32, np.float64)):
                total += 4
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, np.ndarray):
                        total += item.nbytes
                    elif isinstance(item, list):
                        total += sum(
                            i.nbytes if isinstance(i, np.ndarray) else 4 for i in item
                        )
                    elif isinstance(item, dict):
                        total += self._compressed_bytes(item)
                    else:
                        total += 4
            elif isinstance(v, dict):
                total += self._compressed_bytes(v)
        return total

    def evaluate(self, tensor: np.ndarray, **kwargs) -> Dict[str, float]:
        t0 = time.time()
        data, meta = self.compress(tensor, **kwargs)
        t_compress = time.time() - t0

        t0 = time.time()
        recon = self.decompress(data, meta)
        t_decompress = time.time() - t0

        orig = tensor.astype(np.float64).ravel()
        rec = recon.astype(np.float64).ravel()

        orig_bytes = tensor.nbytes
        comp_bytes = self._compressed_bytes(data)

        mse = float(np.mean((orig - rec) ** 2))
        frob_rel = float(np.linalg.norm(orig - rec) / (np.linalg.norm(orig) + 1e-30))
        snr = 10 * np.log10(np.sum(orig**2) / (np.sum((orig - rec) ** 2) + 1e-30))
        cos_sim = float(
            np.dot(orig, rec) / (np.linalg.norm(orig) * np.linalg.norm(rec) + 1e-30)
        )
        rel_err = float(np.mean(np.abs(orig - rec) / (np.abs(orig) + 1e-10)))

        return {
            "method": self.METHOD_NAME,
            "shape": str(tensor.shape),
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio": orig_bytes / max(comp_bytes, 1),
            "mse": mse,
            "frob_rel_error": frob_rel,
            "snr_db": snr,
            "cosine_sim": cos_sim,
            "rel_error_pct": rel_err * 100,
            "t_compress_s": t_compress,
            "t_decompress_s": t_decompress,
        }


def _int8_quantize_per_tensor(x: np.ndarray) -> Tuple[np.ndarray, float]:
    amax = float(np.max(np.abs(x)))
    scale = amax / 127.0 if amax > 1e-8 else 1e-8
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


def _int4_quantize_per_group(
    x: np.ndarray, group_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    gs = min(group_size, n)
    n_groups = (n + gs - 1) // gs
    padded = np.zeros(n_groups * gs, dtype=np.float32)
    padded[:n] = x.astype(np.float32)
    groups = padded.reshape(n_groups, gs)
    amax = np.max(np.abs(groups), axis=1, keepdims=True)
    scales = np.where(amax < 1e-8, 1e-8, amax / 7.0)
    q = np.clip(np.round(groups / scales), -8, 7).astype(np.int8)
    return q, scales.ravel().astype(np.float32)


def _fwht_2d_cols(matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=np.float32)
    for j in range(matrix.shape[1]):
        result[:, j] = fwht(matrix[:, j].astype(np.float32), normalize=normalize)
    return result


def _dct_1d(x: np.ndarray) -> np.ndarray:
    return dct(x.astype(np.float32))


def _idct_1d(x: np.ndarray) -> np.ndarray:
    return idct(x.astype(np.float32))


def _haar_forward_1d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n % 2 == 1:
        x = np.append(x, 0.0)
    even, odd = x[0::2], x[1::2]
    approx = (even + odd) * 0.5
    detail = (even - odd) * 0.5
    return approx, detail


def _haar_inverse_1d(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    approx = np.asarray(approx, dtype=np.float64)
    detail = np.asarray(detail, dtype=np.float64)
    n = len(approx)
    out = np.empty(2 * n, dtype=np.float64)
    out[0::2] = approx + detail
    out[1::2] = approx - detail
    return out


class RowWiseHadamardINT8(NNWeightTransform):
    """Apply Hadamard to each row independently, then INT8 quantize."""

    METHOD_NAME = "row_hadamard_int8"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape
        padded_cols = next_power_of_two(cols)

        rng = np.random.RandomState(self.seed)
        signs = rng.choice([-1, 1], size=(rows, padded_cols)).astype(np.float32)

        H = np.zeros((rows, padded_cols), dtype=np.float32)
        H[:, :cols] = W
        H *= signs
        H = fwht(H, normalize=True)

        scales = np.max(np.abs(H), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(H / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig_cols": np.int32(cols),
            "padded_cols": np.int32(padded_cols),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        seed = int(data["seed"])
        rows = data["quantized"].shape[0]
        padded_cols = int(data["padded_cols"])
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(rows, padded_cols)).astype(np.float32)

        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        q = fwht(q, normalize=True)
        q *= signs
        n_cols = int(data["n_orig_cols"])
        return q[:, :n_cols].reshape(metadata["orig_shape"])


class ColumnWiseHadamardINT8(NNWeightTransform):
    """Apply Hadamard to each column independently, then INT8 quantize."""

    METHOD_NAME = "col_hadamard_int8"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape
        padded_rows = next_power_of_two(rows)

        rng = np.random.RandomState(self.seed)
        signs = rng.choice([-1, 1], size=(padded_rows, cols)).astype(np.float32)

        H = np.zeros((padded_rows, cols), dtype=np.float32)
        H[:rows, :] = W
        H *= signs
        for j in range(cols):
            H[:, j] = fwht(H[:, j], normalize=True)

        scales = np.max(np.abs(H), axis=0, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(H / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig_rows": np.int32(rows),
            "padded_rows": np.int32(padded_rows),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        seed = int(data["seed"])
        padded_rows = int(data["padded_rows"])
        cols = data["quantized"].shape[1]
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(padded_rows, cols)).astype(np.float32)

        q = data["quantized"].astype(np.float32) * data["scales"][np.newaxis, :]
        for j in range(cols):
            q[:, j] = fwht(q[:, j], normalize=True)
        q *= signs
        n_rows = int(data["n_orig_rows"])
        return q[:n_rows, :].reshape(metadata["orig_shape"])


class BlockHadamardINT8(NNWeightTransform):
    """Hadamard in fixed-size blocks, then INT8 per block."""

    METHOD_NAME = "block_hadamard_int8"

    def __init__(self, block_size: int = 128, seed: int = 42):
        self.block_size = block_size
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        bs = self.block_size
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat

        rng = np.random.RandomState(self.seed)
        signs = rng.choice([-1, 1], size=(n_blocks, bs)).astype(np.float32)

        blocks = padded.reshape(n_blocks, bs)
        blocks = blocks * signs
        blocks = fwht(blocks, normalize=True)

        scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks = q.shape[0]
        bs = q.shape[1]
        rng = np.random.RandomState(int(data["seed"]))
        signs = rng.choice([-1, 1], size=(n_blocks, bs)).astype(np.float32)
        q = fwht(q, normalize=True)
        q = q * signs
        flat = q.ravel()
        n = int(data["n_orig"])
        return flat[:n].reshape(metadata["orig_shape"])


class HadamardRowINT4(NNWeightTransform):
    """Hadamard per row + INT4 quantization."""

    METHOD_NAME = "hadamard_row_int4"

    def __init__(self, group_size: int = 128, seed: int = 42):
        self.group_size = group_size
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape
        padded_cols = next_power_of_two(cols)

        rng = np.random.RandomState(self.seed)
        signs = rng.choice([-1, 1], size=(rows, padded_cols)).astype(np.float32)

        H = np.zeros((rows, padded_cols), dtype=np.float32)
        H[:, :cols] = W
        H *= signs
        H = fwht(H, normalize=True)

        flat = H.ravel()
        q, scales = _int4_quantize_per_group(flat, self.group_size)

        data = {
            "quantized": q,
            "scales": scales,
            "n_orig_cols": np.int32(cols),
            "padded_cols": np.int32(padded_cols),
            "n_rows": np.int32(rows),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        seed = int(data["seed"])
        rows = int(data["n_rows"])
        padded_cols = int(data["padded_cols"])
        n = rows * padded_cols

        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(rows, padded_cols)).astype(np.float32)

        groups = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        flat = groups.ravel()[:n]
        H = flat.reshape(rows, padded_cols)
        H = fwht(H, normalize=True)
        H *= signs
        n_cols = int(data["n_orig_cols"])
        return H[:, :n_cols].reshape(metadata["orig_shape"])


class HadamardBlockINT4(NNWeightTransform):
    """Hadamard in blocks of 128 + INT4 per block."""

    METHOD_NAME = "hadamard_block_int4"

    def __init__(self, block_size: int = 128, seed: int = 42):
        self.block_size = block_size
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        bs = self.block_size
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat

        rng = np.random.RandomState(self.seed)
        signs = rng.choice([-1, 1], size=(n_blocks, bs)).astype(np.float32)

        blocks = padded.reshape(n_blocks, bs)
        blocks = blocks * signs
        blocks = fwht(blocks, normalize=True)

        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(amax < 1e-8, 1e-8, amax / 7.0)
        q = np.clip(np.round(blocks / scales), -8, 7).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks = q.shape[0]
        bs = q.shape[1]
        rng = np.random.RandomState(int(data["seed"]))
        signs = rng.choice([-1, 1], size=(n_blocks, bs)).astype(np.float32)
        q = fwht(q, normalize=True)
        q = q * signs
        flat = q.ravel()
        n = int(data["n_orig"])
        return flat[:n].reshape(metadata["orig_shape"])


class DCTRowINT8(NNWeightTransform):
    """DCT per row, keep all coefficients, INT8 quantize."""

    METHOD_NAME = "dct_row_int8"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape

        H = np.zeros_like(W)
        for i in range(rows):
            H[i] = _dct_1d(W[i])

        scales = np.max(np.abs(H), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(H / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        rows = q.shape[0]
        result = np.zeros_like(q)
        for i in range(rows):
            result[i] = _idct_1d(q[i])
        return result.reshape(metadata["orig_shape"])


class DCTBlockINT8(NNWeightTransform):
    """DCT per block, then INT8."""

    METHOD_NAME = "dct_block_int8"

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        bs = self.block_size
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat

        blocks = padded.reshape(n_blocks, bs)
        dct_blocks = np.zeros_like(blocks)
        for i in range(n_blocks):
            dct_blocks[i] = _dct_1d(blocks[i])

        scales = np.max(np.abs(dct_blocks), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(dct_blocks / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks, bs = q.shape
        recon = np.zeros_like(q)
        for i in range(n_blocks):
            recon[i] = _idct_1d(q[i])
        flat = recon.ravel()
        n = int(data["n_orig"])
        return flat[:n].reshape(metadata["orig_shape"])


class WaveletRowINT8(NNWeightTransform):
    """Haar wavelet per row (3 levels), INT8 on all coefficients."""

    METHOD_NAME = "wavelet_row_int8"

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape

        all_q = []
        all_scales = []
        all_lengths = []
        final_approx_list = []
        n_packed_list = []

        for i in range(rows):
            current = W[i].astype(np.float64)
            detail_levels = []
            for _ in range(self.n_levels):
                if len(current) <= 2:
                    break
                approx, detail = _haar_forward_1d(current.astype(np.float32))
                detail_levels.append(detail)
                current = approx.astype(np.float64)

            lengths = [len(d) for d in detail_levels]
            packed = np.concatenate(
                [current] + [d.astype(np.float64) for d in reversed(detail_levels)]
            )
            n_packed = len(packed)

            amax = float(np.max(np.abs(packed)))
            scale = amax / 127.0 if amax > 1e-8 else 1e-8
            q = np.clip(np.round(packed / scale), -128, 127).astype(np.int8)

            all_q.append(q)
            all_scales.append(scale)
            all_lengths.append(lengths)
            final_approx_list.append(len(current))
            n_packed_list.append(n_packed)

        data = {
            "quantized": all_q,
            "scales": np.array(all_scales, dtype=np.float32),
            "all_lengths": all_lengths,
            "final_approx_lens": np.array(final_approx_list, dtype=np.int32),
            "n_packed": np.array(n_packed_list, dtype=np.int32),
            "n_rows": np.int32(rows),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows = int(data["n_rows"])
        cols = metadata["orig_shape"][1]
        result = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            q = data["quantized"][i].astype(np.float32) * data["scales"][i]
            lengths = data["all_lengths"][i]
            n_packed = int(data["n_packed"][i])
            final_len = int(data["final_approx_lens"][i])

            q = q[:n_packed]
            final_approx = q[:final_len].astype(np.float64)
            detail_coeffs = q[final_len:].astype(np.float64)

            current = final_approx
            offset = 0
            for length in reversed(lengths):
                detail = detail_coeffs[offset : offset + length]
                offset += length
                current = _haar_inverse_1d(current[:length], detail[:length])

            result[i, : min(len(current), cols)] = current[:cols]

        return result


class WaveletBlockINT8(NNWeightTransform):
    """Haar wavelet in blocks, INT8."""

    METHOD_NAME = "wavelet_block_int8"

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        bs = self.block_size
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat

        blocks = padded.reshape(n_blocks, bs)
        approx = np.zeros((n_blocks, bs // 2), dtype=np.float64)
        detail = np.zeros((n_blocks, bs // 2), dtype=np.float64)
        for i in range(n_blocks):
            a, d = _haar_forward_1d(blocks[i])
            approx[i] = a[: bs // 2]
            detail[i] = d[: bs // 2]

        all_coeffs = np.concatenate([approx, detail], axis=1)
        scales = np.max(np.abs(all_coeffs), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(all_coeffs / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks, bs = q.shape
        half = bs // 2
        approx = q[:, :half].astype(np.float64)
        detail = q[:, half:].astype(np.float64)

        recon = np.zeros((n_blocks, bs), dtype=np.float64)
        for i in range(n_blocks):
            inv = _haar_inverse_1d(approx[i], detail[i])
            recon[i] = inv[:bs]

        flat = recon.ravel()
        n = int(data["n_orig"])
        return flat[:n].astype(np.float32).reshape(metadata["orig_shape"])


class HadamardDCTINT8(NNWeightTransform):
    """Hadamard first, then DCT on blocks, then INT8."""

    METHOD_NAME = "hadamard_dct_int8"

    def __init__(self, block_size: int = 128, seed: int = 42):
        self.block_size = block_size
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        padded_n = next_power_of_two(n)

        rng = np.random.RandomState(self.seed)
        signs = rng.choice([-1, 1], size=padded_n).astype(np.float32)

        buf = np.zeros(padded_n, dtype=np.float32)
        buf[:n] = flat
        buf *= signs
        buf = fwht(buf, normalize=True)

        bs = self.block_size
        n_blocks = padded_n // bs
        blocks = buf[: n_blocks * bs].reshape(n_blocks, bs)
        dct_blocks = np.zeros_like(blocks)
        for i in range(n_blocks):
            dct_blocks[i] = _dct_1d(blocks[i])

        scales = np.max(np.abs(dct_blocks), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(dct_blocks / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
            "padded_n": np.int32(padded_n),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks, bs = q.shape
        dct_recon = np.zeros_like(q)
        for i in range(n_blocks):
            dct_recon[i] = _idct_1d(q[i])

        buf = dct_recon.ravel()
        padded_n = int(data["padded_n"])
        if len(buf) < padded_n:
            buf = np.pad(buf, (0, padded_n - len(buf)))
        buf = fwht(buf, normalize=True)

        seed = int(data["seed"])
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=padded_n).astype(np.float32)
        buf *= signs
        n = int(data["n_orig"])
        return buf[:n].reshape(metadata["orig_shape"])


class MultiScaleINT8(NNWeightTransform):
    """Wavelet decomposition to 3 levels on each row, INT8 on all coefficients."""

    METHOD_NAME = "multiscale_int8"

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape

        all_q = []
        all_scales = []
        all_lengths = []
        final_lens = []
        packed_lens = []

        for i in range(rows):
            current = W[i].astype(np.float64)
            detail_levels = []
            for _ in range(self.n_levels):
                if len(current) <= 2:
                    break
                approx, detail = _haar_forward_1d(current.astype(np.float32))
                detail_levels.append(detail)
                current = approx.astype(np.float64)

            lengths = [len(d) for d in detail_levels]
            packed = np.concatenate(
                [current] + [d.astype(np.float64) for d in reversed(detail_levels)]
            )

            amax = float(np.max(np.abs(packed)))
            scale = amax / 127.0 if amax > 1e-8 else 1e-8
            q = np.clip(np.round(packed / scale), -128, 127).astype(np.int8)

            all_q.append(q)
            all_scales.append(scale)
            all_lengths.append(lengths)
            final_lens.append(len(current))
            packed_lens.append(len(packed))

        data = {
            "quantized": all_q,
            "scales": np.array(all_scales, dtype=np.float32),
            "all_lengths": all_lengths,
            "final_lens": np.array(final_lens, dtype=np.int32),
            "packed_lens": np.array(packed_lens, dtype=np.int32),
            "n_rows": np.int32(rows),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows = int(data["n_rows"])
        cols = metadata["orig_shape"][1]
        result = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            q = data["quantized"][i].astype(np.float32) * data["scales"][i]
            lengths = data["all_lengths"][i]
            n_packed = int(data["packed_lens"][i])
            final_len = int(data["final_lens"][i])

            q = q[:n_packed]
            final_approx = q[:final_len].astype(np.float64)
            detail_coeffs = q[final_len:].astype(np.float64)

            current = final_approx
            offset = 0
            for length in reversed(lengths):
                detail = detail_coeffs[offset : offset + length]
                offset += length
                current = _haar_inverse_1d(current[:length], detail[:length])

            result[i, : min(len(current), cols)] = current[:cols]

        return result


class MixedResolutionINT8(NNWeightTransform):
    """Low-frequency (approx) -> INT8, high-frequency (detail) -> INT4."""

    METHOD_NAME = "mixed_resolution_int8"

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape

        all_data_rows = []
        for i in range(rows):
            current = W[i].astype(np.float64)
            detail_levels = []
            for _ in range(self.n_levels):
                if len(current) <= 2:
                    break
                approx, detail = _haar_forward_1d(current.astype(np.float32))
                detail_levels.append(detail)
                current = approx.astype(np.float64)

            approx_arr = current.astype(np.float32)
            a_max = float(np.max(np.abs(approx_arr))) if len(approx_arr) > 0 else 1e-8
            a_scale = a_max / 127.0 if a_max > 1e-8 else 1e-8
            a_q = np.clip(np.round(approx_arr / a_scale), -128, 127).astype(np.int8)

            detail_q_all = []
            detail_scales_all = []
            for d in reversed(detail_levels):
                d_arr = d.astype(np.float32)
                d_max = float(np.max(np.abs(d_arr)))
                d_scale = d_max / 7.0 if d_max > 1e-8 else 1e-8
                d_q = np.clip(np.round(d_arr / d_scale), -8, 7).astype(np.int8)
                detail_q_all.append(d_q)
                detail_scales_all.append(d_scale)

            lengths = [len(d) for d in reversed(detail_levels)]
            total_detail = sum(lengths)

            detail_q_packed = (
                np.concatenate(detail_q_all)
                if detail_q_all
                else np.array([], dtype=np.int8)
            )
            detail_scales_packed = np.array(detail_scales_all, dtype=np.float32)

            all_data_rows.append(
                {
                    "a_q": a_q,
                    "a_scale": a_scale,
                    "d_q": detail_q_packed,
                    "d_scales": detail_scales_packed,
                    "lengths": lengths,
                    "total_detail": total_detail,
                }
            )

        data = {
            "rows_data": all_data_rows,
            "n_rows": np.int32(rows),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows = int(data["n_rows"])
        cols = metadata["orig_shape"][1]
        result = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            rd = data["rows_data"][i]
            current = rd["a_q"].astype(np.float64) * rd["a_scale"]
            lengths = rd["lengths"]

            d_q = rd["d_q"].astype(np.float64)
            d_scales = rd["d_scales"]

            offset = 0
            for j, length in enumerate(lengths):
                detail = d_q[offset : offset + length] * d_scales[j]
                offset += length
                current = _haar_inverse_1d(current[:length], detail[:length])

            result[i, : min(len(current), cols)] = current[:cols]

        return result


class OutlierPreservingINT8(NNWeightTransform):
    """Detect outlier rows (by L2 norm), store at FP32, INT8 on rest."""

    METHOD_NAME = "outlier_preserving_int8"

    def __init__(self, outlier_percentile: float = 95):
        self.outlier_percentile = outlier_percentile

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape

        row_norms = np.linalg.norm(W, axis=1)
        threshold = np.percentile(row_norms, self.outlier_percentile)
        outlier_mask = row_norms > threshold

        outlier_rows = W[outlier_mask].copy()
        normal_rows = W[~outlier_mask]

        if len(normal_rows) > 0:
            amax = np.max(np.abs(normal_rows), axis=1, keepdims=True)
            scales = np.where(amax < 1e-8, 1e-8, amax / 127.0)
            q_normal = np.clip(np.round(normal_rows / scales), -128, 127).astype(
                np.int8
            )
        else:
            q_normal = np.array([], dtype=np.int8).reshape(0, cols)
            scales = np.array([], dtype=np.float32)

        data = {
            "normal_q": q_normal,
            "normal_scales": scales.ravel().astype(np.float32),
            "outlier_rows": outlier_rows,
            "outlier_indices": np.where(outlier_mask)[0].astype(np.int32),
            "n_rows": np.int32(rows),
            "n_cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        rows = int(data["n_rows"])
        cols = int(data["n_cols"])
        result = np.zeros((rows, cols), dtype=np.float32)

        q = data["normal_q"].astype(np.float32) * data["normal_scales"][:, np.newaxis]
        normal_mask = np.ones(rows, dtype=bool)
        outlier_idx = data["outlier_indices"]
        normal_mask[outlier_idx] = False

        result[normal_mask] = q

        if len(outlier_idx) > 0:
            result[outlier_idx] = data["outlier_rows"]

        return result


class SensitivityWeightedINT8(NNWeightTransform):
    """Profile each block's sensitivity, allocate tighter quantization to high-sensitivity blocks."""

    METHOD_NAME = "sensitivity_weighted_int8"

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        gs = min(self.group_size, n)
        n_groups = (n + gs - 1) // gs
        padded = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat

        blocks = padded.reshape(n_groups, gs)

        block_energy = np.sum(blocks**2, axis=1)
        max_energy = np.max(block_energy)
        sensitivity = block_energy / (max_energy + 1e-8)

        scale_factor = 1.0 - 0.05 * sensitivity
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        base_scales = np.where(amax < 1e-8, 1e-8, amax / 127.0)
        effective_scale = base_scales * scale_factor[:, np.newaxis]
        effective_scale = np.where(effective_scale < 1e-8, 1e-8, effective_scale)

        q = np.clip(np.round(blocks / effective_scale), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": effective_scale.ravel().astype(np.float32),
            "n_orig": np.int32(n),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        flat = q.ravel()
        n = int(data["n_orig"])
        return flat[:n].reshape(metadata["orig_shape"])


class BlockSkewHadamardINT8(NNWeightTransform):
    """Randomized Hadamard: each block gets independent random sign flip."""

    METHOD_NAME = "block_skew_hadamard_int8"

    def __init__(self, block_size: int = 128, seed: int = 42):
        self.block_size = block_size
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        bs = self.block_size
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat

        rng = np.random.RandomState(self.seed)
        block_signs = np.zeros((n_blocks, bs), dtype=np.float32)
        for i in range(n_blocks):
            block_signs[i] = rng.choice([-1, 1], size=bs).astype(np.float32)

        blocks = padded.reshape(n_blocks, bs)
        blocks = blocks * block_signs
        blocks = fwht(blocks, normalize=True)

        scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks, bs = q.shape
        rng = np.random.RandomState(int(data["seed"]))
        block_signs = np.zeros((n_blocks, bs), dtype=np.float32)
        for i in range(n_blocks):
            block_signs[i] = rng.choice([-1, 1], size=bs).astype(np.float32)
        q = fwht(q, normalize=True)
        q = q * block_signs
        flat = q.ravel()
        n = int(data["n_orig"])
        return flat[:n].reshape(metadata["orig_shape"])


class RandomRotationINT8(NNWeightTransform):
    """Random orthogonal rotation + INT8."""

    METHOD_NAME = "random_rotation_int8"

    def __init__(self, block_size: int = 256, seed: int = 42):
        self.block_size = block_size
        self.seed = seed

    def _random_orthogonal(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        Q = np.eye(n, dtype=np.float32)
        for _ in range(3):
            v = rng.randn(n).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-10)
            H = np.eye(n, dtype=np.float32) - 2.0 * np.outer(v, v)
            Q = Q @ H
        return Q

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        bs = min(self.block_size, n)
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat

        blocks = padded.reshape(n_blocks, bs)
        rotated = np.zeros_like(blocks)
        for i in range(n_blocks):
            Q = self._random_orthogonal(bs, np.random.RandomState(self.seed + i))
            rotated[i] = blocks[i] @ Q.T

        scales = np.max(np.abs(rotated), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(rotated / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
            "block_size": np.int32(bs),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks, bs = q.shape

        result = np.zeros_like(q)
        for i in range(n_blocks):
            Q = self._random_orthogonal(
                bs, np.random.RandomState(int(data["seed"]) + i)
            )
            result[i] = q[i] @ Q

        flat = result.ravel()
        n = int(data["n_orig"])
        return flat[:n].reshape(metadata["orig_shape"])


class KroneckerHadamardINT8(NNWeightTransform):
    """Kronecker product of small Hadamard matrices for efficient structured decorrelation."""

    METHOD_NAME = "kronecker_hadamard_int8"

    def __init__(self, small_size: int = 8, block_size: int = 128, seed: int = 42):
        self.small_size = small_size
        self.block_size = block_size
        self.seed = seed

    def _hadamard_matrix(self, n: int) -> np.ndarray:
        if n == 1:
            return np.array([[1.0]], dtype=np.float32)
        H_half = self._hadamard_matrix(n // 2)
        return np.block(
            [
                [H_half, H_half],
                [H_half, -H_half],
            ]
        ).astype(np.float32) / np.sqrt(2)

    def _kronecker_hadamard(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        s = self.small_size
        big = s * s
        n_blocks = (n + big - 1) // big
        padded = np.zeros(n_blocks * big, dtype=np.float32)
        padded[:n] = x

        H = self._hadamard_matrix(s)
        blocks = padded.reshape(n_blocks, big)
        result = np.zeros_like(blocks)
        for i in range(n_blocks):
            b = blocks[i].reshape(s, s)
            b = H @ b @ H.T
            result[i] = b.ravel()

        return result.ravel()[:n]

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)

        rng = np.random.RandomState(self.seed)
        signs = rng.choice([-1, 1], size=n).astype(np.float32)

        x = flat * signs
        x = self._kronecker_hadamard(x)

        bs = self.block_size
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = x
        blocks = padded.reshape(n_blocks, bs)

        scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 127.0
        scales = np.where(scales < 1e-8, 1e-8, scales)
        q = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)

        data = {
            "quantized": q,
            "scales": scales.ravel().astype(np.float32),
            "n_orig": np.int32(n),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        x = q.ravel()
        n = int(data["n_orig"])
        x = x[:n]
        x = self._kronecker_hadamard(x)
        rng = np.random.RandomState(int(data["seed"]))
        signs = rng.choice([-1, 1], size=n).astype(np.float32)
        x = x * signs
        return x[:n].reshape(metadata["orig_shape"])


class LowRankHadamardINT8(NNWeightTransform):
    """SVD at high rank + Hadamard + INT8 on factors."""

    METHOD_NAME = "lowrank_hadamard_int8"

    def __init__(self, keep_energy: float = 0.999, seed: int = 42):
        self.keep_energy = keep_energy
        self.seed = seed

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float64)
        rows, cols = W.shape

        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        total_energy = float(np.sum(S**2))
        cumulative = np.cumsum(S**2) / (total_energy + 1e-30)
        k = int(np.searchsorted(cumulative, self.keep_energy)) + 1
        k = max(1, min(k, len(S)))

        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)

        u_flat = U_k.ravel()
        u_scale = float(np.max(np.abs(u_flat)) / 127.0) if np.any(u_flat) else 1e-8
        U_q = np.clip(np.round(u_flat / u_scale), -128, 127).astype(np.int8)

        s_scale = float(np.max(np.abs(S_k)) / 127.0) if np.any(S_k) else 1e-8
        S_q = np.clip(np.round(S_k / s_scale), -128, 127).astype(np.int8)

        v_flat = Vt_k.ravel()
        v_scale = float(np.max(np.abs(v_flat)) / 127.0) if np.any(v_flat) else 1e-8
        V_q = np.clip(np.round(v_flat / v_scale), -128, 127).astype(np.int8)

        data = {
            "U_q": U_q,
            "U_scale": np.float32(u_scale),
            "U_shape": np.array(U_k.shape, dtype=np.int32),
            "S_q": S_q,
            "S_scale": np.float32(s_scale),
            "V_q": V_q,
            "V_scale": np.float32(v_scale),
            "V_shape": np.array(Vt_k.shape, dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        U_shape = tuple(data["U_shape"])
        V_shape = tuple(data["V_shape"])

        U = (
            (data["U_q"].astype(np.float32) * data["U_scale"])
            .reshape(U_shape)
            .astype(np.float64)
        )
        S = (data["S_q"].astype(np.float32) * data["S_scale"]).astype(np.float64)
        V = (
            (data["V_q"].astype(np.float32) * data["V_scale"])
            .reshape(V_shape)
            .astype(np.float64)
        )

        W = U @ np.diag(S) @ V
        return W.astype(np.float32).reshape(metadata["orig_shape"])


class BlockDiagonalINT8(NNWeightTransform):
    """Partition matrix into blocks, INT8 quantize independently per block."""

    METHOD_NAME = "block_diagonal_int8"

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        W = tensor.astype(np.float32)
        rows, cols = W.shape
        bs = self.block_size

        r_pad = (bs - rows % bs) % bs
        c_pad = (bs - cols % bs) % bs
        W_padded = np.zeros((rows + r_pad, cols + c_pad), dtype=np.float32)
        W_padded[:rows, :cols] = W

        pr, pc = W_padded.shape
        n_r_blocks = pr // bs
        n_c_blocks = pc // bs

        all_blocks = []
        all_scales = []

        for i in range(n_r_blocks):
            for j in range(n_c_blocks):
                block = W_padded[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs]
                flat = block.ravel()
                amax = float(np.max(np.abs(flat)))
                s = amax / 127.0 if amax > 1e-8 else 1e-8
                q = np.clip(np.round(flat / s), -128, 127).astype(np.int8)
                all_blocks.append(q)
                all_scales.append(s)

        data = {
            "quantized": np.array(all_blocks, dtype=np.int8),
            "scales": np.array(all_scales, dtype=np.float32),
            "n_r_blocks": np.int32(n_r_blocks),
            "n_c_blocks": np.int32(n_c_blocks),
            "block_size": np.int32(bs),
            "orig_rows": np.int32(rows),
            "orig_cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n_r = int(data["n_r_blocks"])
        n_c = int(data["n_c_blocks"])
        bs = int(data["block_size"])
        rows = int(data["orig_rows"])
        cols = int(data["orig_cols"])

        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        result = np.zeros((n_r * bs, n_c * bs), dtype=np.float32)

        idx = 0
        for i in range(n_r):
            for j in range(n_c):
                result[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = q[idx].reshape(
                    bs, bs
                )
                idx += 1

        return result[:rows, :cols].reshape(metadata["orig_shape"])


class AdaptiveTransformINT8(NNWeightTransform):
    """For each block, try Hadamard, DCT, Wavelet, pick the one with lowest INT8 error."""

    METHOD_NAME = "adaptive_transform_int8"

    def __init__(self, block_size: int = 128, seed: int = 42):
        self.block_size = block_size
        self.seed = seed

    def _apply_and_measure(
        self, block: np.ndarray, transform_id: int
    ) -> Tuple[np.ndarray, float, float]:
        x = block.astype(np.float32)
        bs = len(x)

        if transform_id == 0:
            rng = np.random.RandomState(self.seed)
            signs = rng.choice([-1, 1], size=bs).astype(np.float32)
            x_t = fwht(x * signs, normalize=True)
        elif transform_id == 1:
            x_t = _dct_1d(x)
        elif transform_id == 2:
            a, d = _haar_forward_1d(x)
            x_t = np.concatenate([a, d])
        else:
            x_t = x.copy()

        amax = float(np.max(np.abs(x_t)))
        scale = amax / 127.0 if amax > 1e-8 else 1e-8
        q = np.clip(np.round(x_t / scale), -128, 127).astype(np.int8)
        deq = q.astype(np.float32) * scale

        if transform_id == 0:
            rng = np.random.RandomState(self.seed)
            signs = rng.choice([-1, 1], size=bs).astype(np.float32)
            inv = fwht(deq, normalize=True) * signs
        elif transform_id == 1:
            inv = _idct_1d(deq)
        elif transform_id == 2:
            half = len(x_t) // 2
            inv = _haar_inverse_1d(
                deq[:half].astype(np.float64), deq[half:].astype(np.float64)
            )
            inv = inv[:bs]
        else:
            inv = deq

        error = float(np.sum((block.astype(np.float32) - inv) ** 2))
        return q, scale, error

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        bs = self.block_size
        n_blocks = (n + bs - 1) // bs
        padded = np.zeros(n_blocks * bs, dtype=np.float32)
        padded[:n] = flat

        blocks = padded.reshape(n_blocks, bs)
        all_q = []
        all_scales = []
        all_transform_ids = []

        for i in range(n_blocks):
            best_q, best_scale, best_error, best_id = None, 0, float("inf"), 0
            for tid in range(3):
                q, scale, error = self._apply_and_measure(blocks[i], tid)
                if error < best_error:
                    best_q, best_scale, best_error, best_id = q, scale, error, tid
            all_q.append(best_q)
            all_scales.append(best_scale)
            all_transform_ids.append(best_id)

        data = {
            "quantized": np.array(all_q, dtype=np.int8),
            "scales": np.array(all_scales, dtype=np.float32),
            "transform_ids": np.array(all_transform_ids, dtype=np.int8),
            "n_orig": np.int32(n),
            "block_size": np.int32(bs),
            "seed": np.int32(self.seed),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        q = data["quantized"].astype(np.float32) * data["scales"][:, np.newaxis]
        n_blocks, bs = q.shape
        seed = int(data["seed"])

        result = np.zeros((n_blocks, bs), dtype=np.float32)
        for i in range(n_blocks):
            recon = q[i]
            tid = int(data["transform_ids"][i])

            if tid == 0:
                rng = np.random.RandomState(seed)
                signs = rng.choice([-1, 1], size=bs).astype(np.float32)
                inv = fwht(recon, normalize=True) * signs
            elif tid == 1:
                inv = _idct_1d(recon)
            elif tid == 2:
                half = bs // 2
                inv = _haar_inverse_1d(
                    recon[:half].astype(np.float64), recon[half:].astype(np.float64)
                )
                inv = inv[:bs].astype(np.float32)
            else:
                inv = recon

            result[i] = inv[:bs]

        flat = result.ravel()
        n = int(data["n_orig"])
        return flat[:n].reshape(metadata["orig_shape"])


ALL_NN_TRANSFORMS = {
    "row_hadamard_int8": RowWiseHadamardINT8,
    "col_hadamard_int8": ColumnWiseHadamardINT8,
    "block_hadamard_int8": BlockHadamardINT8,
    "hadamard_row_int4": HadamardRowINT4,
    "hadamard_block_int4": HadamardBlockINT4,
    "dct_row_int8": DCTRowINT8,
    "dct_block_int8": DCTBlockINT8,
    "wavelet_row_int8": WaveletRowINT8,
    "wavelet_block_int8": WaveletBlockINT8,
    "hadamard_dct_int8": HadamardDCTINT8,
    "multiscale_int8": MultiScaleINT8,
    "mixed_resolution_int8": MixedResolutionINT8,
    "outlier_preserving_int8": OutlierPreservingINT8,
    "sensitivity_weighted_int8": SensitivityWeightedINT8,
    "block_skew_hadamard_int8": BlockSkewHadamardINT8,
    "random_rotation_int8": RandomRotationINT8,
    "kronecker_hadamard_int8": KroneckerHadamardINT8,
    "lowrank_hadamard_int8": LowRankHadamardINT8,
    "block_diagonal_int8": BlockDiagonalINT8,
    "adaptive_transform_int8": AdaptiveTransformINT8,
}

"""Cross-layer delta coding — compress layer N as a sparse delta from layer N-1."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CrossLayerDeltaCompression:
    """Compresses layers by modeling layer N as a sparse delta from layer N-1.

    Layer 0 is fully compressed with randomized SVD.
    Subsequent layers store only the top-k coefficients of (layer_N - layer_N-1).
    """

    name = "cross_layer_delta"
    category = "novel"

    def __init__(self, rank: int = 32, topk_frac: float = 0.1):
        self.rank = rank
        self.topk_frac = topk_frac

    def compress(
        self,
        tensor: np.ndarray,
        prev_tensor: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[bytes, dict]:
        if prev_tensor is None:
            return self._compress_base(tensor, **kwargs)
        return self._compress_delta(tensor, prev_tensor, **kwargs)

    def decompress(
        self, data: bytes, metadata: dict, prev_tensor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        mode = metadata.get("mode", "base")
        if mode == "delta":
            return self._decompress_delta(data, metadata, prev_tensor)
        return self._decompress_base(data, metadata)

    def _compress_base(
        self, tensor: np.ndarray, rank: Optional[int] = None, **kwargs
    ) -> Tuple[bytes, dict]:
        r = rank or self.rank
        mat = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = mat.shape
        k = min(r, min(m, n) - 1)
        rng = np.random.RandomState(42)
        O = rng.randn(n, k + 10).astype(mat.dtype)
        Y = mat @ O
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ mat
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = (Q @ U_hat[:, :k]).astype(np.float32)
        S = S[:k].astype(np.float32)
        Vt = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += U.tobytes() + S.tobytes() + Vt.tobytes()
        return bytes(buf), {
            "mode": "base",
            "shape": tensor.shape,
            "m": m,
            "n": n,
            "k": k,
            "rank": r,
        }

    def _decompress_base(self, data: bytes, metadata: dict) -> np.ndarray:
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        U = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        S = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(k, n)
        result = (U * S) @ Vt
        shape = metadata.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result

    def _compress_delta(
        self, tensor: np.ndarray, prev_tensor: np.ndarray, **kwargs
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        prev_flat = prev_tensor.ravel().astype(np.float32)
        if len(prev_flat) != len(flat):
            prev_flat = np.resize(prev_flat, len(flat))
        delta = flat - prev_flat
        n = len(delta)
        k = max(1, int(n * self.topk_frac))
        delta_abs = np.abs(delta)
        threshold = float(np.partition(delta_abs, n - k)[n - k])
        mask = delta_abs >= threshold
        indices = np.where(mask)[0].astype(np.int32)
        values = delta[mask].astype(np.float32)
        n_indices = len(indices)
        buf = struct.pack("<II", n, n_indices)
        buf += indices.tobytes() + values.tobytes()
        return bytes(buf), {
            "mode": "delta",
            "shape": tensor.shape,
            "n": n,
            "topk_frac": self.topk_frac,
            "n_indices": n_indices,
        }

    def _decompress_delta(
        self, data: bytes, metadata: dict, prev_tensor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n, n_indices = struct.unpack_from("<II", data, 0)
        pos = 8
        indices = np.frombuffer(data[pos : pos + n_indices * 4], dtype=np.int32)
        pos += n_indices * 4
        values = np.frombuffer(data[pos : pos + n_indices * 4], dtype=np.float32)
        delta = np.zeros(n, dtype=np.float32)
        delta[indices] = values
        if prev_tensor is not None:
            prev_flat = prev_tensor.ravel().astype(np.float32)
            if len(prev_flat) != n:
                prev_flat = np.resize(prev_flat, n)
            result = prev_flat + delta
        else:
            result = delta
        shape = metadata.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result

    def compress_layers(
        self, tensors: List[np.ndarray], **kwargs
    ) -> List[Tuple[bytes, dict]]:
        results = []
        prev = None
        for tensor in tensors:
            data, meta = self.compress(tensor, prev_tensor=prev, **kwargs)
            results.append((data, meta))
            recon = self.decompress(data, meta, prev_tensor=prev)
            prev = recon
        return results

    def decompress_layers(
        self, compressed: List[Tuple[bytes, dict]]
    ) -> List[np.ndarray]:
        results = []
        prev = None
        for data, meta in compressed:
            recon = self.decompress(data, meta, prev_tensor=prev)
            results.append(recon)
            prev = recon
        return results


class BlockwiseCrossLayerDelta:
    """Block-wise cross-layer delta coding with per-block SVD base."""

    name = "blockwise_cross_layer_delta"
    category = "novel"

    def __init__(self, block_size: int = 256, rank: int = 16, topk_frac: float = 0.15):
        self.block_size = block_size
        self.rank = rank
        self.topk_frac = topk_frac

    def compress(
        self,
        tensor: np.ndarray,
        prev_tensor: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[bytes, dict]:
        if prev_tensor is None:
            return self._compress_base(tensor, **kwargs)
        return self._compress_delta(tensor, prev_tensor, **kwargs)

    def decompress(
        self, data: bytes, metadata: dict, prev_tensor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        mode = metadata.get("mode", "base")
        if mode == "delta":
            return self._decompress_delta(data, metadata, prev_tensor)
        return self._decompress_base(data, metadata)

    def _compress_base(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        from .. import BlockINT8Wrapper

        return BlockINT8Wrapper.compress(tensor, block_size=self.block_size)

    def _decompress_base(self, data: bytes, metadata: dict) -> np.ndarray:
        from .. import BlockINT8Wrapper

        return BlockINT8Wrapper.decompress(data, metadata)

    def _compress_delta(
        self, tensor: np.ndarray, prev_tensor: np.ndarray, **kwargs
    ) -> Tuple[bytes, dict]:
        base = CrossLayerDeltaCompression(rank=self.rank, topk_frac=self.topk_frac)
        return base._compress_delta(tensor, prev_tensor)

    def _decompress_delta(
        self, data: bytes, metadata: dict, prev_tensor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        base = CrossLayerDeltaCompression(rank=self.rank, topk_frac=self.topk_frac)
        return base._decompress_delta(data, metadata, prev_tensor)


class SparseDeltaEncoding:
    """Sparse delta encoding for compressing residuals between layers.

    Only stores the top-k% largest coefficients of the delta matrix.
    """

    name = "sparse_delta"
    category = "novel"

    def __init__(self, keep_ratio: float = 0.05):
        self.keep_ratio = keep_ratio

    def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, Dict]:
        flat = tensor.flatten()
        k = max(1, int(len(flat) * self.keep_ratio))
        indices = np.argpartition(-np.abs(flat), k)[:k]
        values = flat[indices].astype(np.float16)
        return values.tobytes(), {
            "indices": indices.astype(np.int32).tobytes(),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "keep_ratio": self.keep_ratio,
        }

    def decompress(self, data: bytes, meta: Dict) -> np.ndarray:
        shape = meta["shape"]
        result = np.zeros(shape, dtype=np.float32)
        values = np.frombuffer(data, dtype=np.float16)
        indices = np.frombuffer(meta["indices"], dtype=np.int32)
        result.flat[indices] = values.astype(np.float32)
        return result

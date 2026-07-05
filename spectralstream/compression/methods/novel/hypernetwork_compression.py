"""Hypernetwork/INR compression — generate weight values from a tiny MLP.

Treats the full weight tensor as a signal to be modeled implicitly by a
small coordinate-based MLP. Instead of storing n×m weight values, store
only the ~thousands of MLP parameters.
"""

from __future__ import annotations


import math
import struct
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class HypernetworkMLP:
    """Minimal 2-layer MLP with sinusoidal activations (SIREN-style).

    Maps (layer_id, row, col) → weight value.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, output_dim: int = 1):
        rng = np.random.RandomState(42)
        scale_w = math.sqrt(2.0 / input_dim)
        self.w1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale_w
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        scale_w2 = math.sqrt(2.0 / hidden_dim)
        self.w2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_w2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        scale_out = math.sqrt(2.0 / hidden_dim)
        self.w_out = rng.randn(hidden_dim, output_dim).astype(np.float32) * scale_out
        self.b_out = np.zeros(output_dim, dtype=np.float32)

    def forward(self, coords: np.ndarray) -> np.ndarray:
        h = _silu(coords @ self.w1 + self.b1)
        h = _silu(h @ self.w2 + self.b2)
        return h @ self.w_out + self.b_out

    def forward_siren(self, coords: np.ndarray) -> np.ndarray:
        w0 = 30.0
        h = np.sin(w0 * (coords @ self.w1 + self.b1))
        h = np.sin(h @ self.w2 + self.b2)
        return h @ self.w_out + self.b_out

    def get_params(self) -> bytes:
        buf = bytearray()
        for arr in [self.w1, self.b1, self.w2, self.b2, self.w_out, self.b_out]:
            buf += arr.astype(np.float32).tobytes()
        return bytes(buf)

    def set_params(self, data: bytes) -> None:
        pos = 0
        shapes = [
            self.w1.shape,
            self.b1.shape,
            self.w2.shape,
            self.b2.shape,
            self.w_out.shape,
            self.b_out.shape,
        ]
        for arr_name, shape in zip(["w1", "b1", "w2", "b2", "w_out", "b_out"], shapes):
            nbytes = int(np.prod(shape)) * 4
            arr = np.frombuffer(data[pos : pos + nbytes], dtype=np.float32).reshape(
                shape
            )
            setattr(self, arr_name, arr)
            pos += nbytes

    def num_params(self) -> int:
        return (
            self.w1.size
            + self.b1.size
            + self.w2.size
            + self.b2.size
            + self.w_out.size
            + self.b_out.size
        )


class HypernetworkCompression:
    """Compresses weight matrices by training a tiny MLP to generate them.

    The MLP maps (block_row, block_col, layer_id) → block_mean weight value.
    Only the MLP weights (~thousands of parameters) are stored instead of
    the full weight matrix.

    For large matrices, breaks the tensor into blocks and learns the block
    pattern rather than individual elements.
    """

    name = "hypernetwork_compress"
    category = "novel"

    def __init__(
        self,
        hidden_dim: int = 32,
        block_size: int = 64,
        learning_rate: float = 0.01,
        max_iter: int = 500,
        use_siren: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.use_siren = use_siren

    def compress(
        self, tensor: np.ndarray, layer_id: int = 0, **kwargs
    ) -> Tuple[bytes, dict]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        n_blocks = max(1, int(math.ceil(n / self.block_size)))

        hidden_dim = kwargs.get("hidden_dim", self.hidden_dim)
        block_size = kwargs.get("block_size", self.block_size)
        lr = kwargs.get("learning_rate", self.learning_rate)
        max_iter = kwargs.get("max_iter", self.max_iter)
        use_siren = kwargs.get("use_siren", self.use_siren)

        block_means = np.zeros(n_blocks, dtype=np.float32)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            block_means[b] = float(np.mean(flat[start:end]))

        mlp = HypernetworkMLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1)

        coords = np.zeros((n_blocks, 2), dtype=np.float32)
        for b in range(n_blocks):
            coords[b, 0] = float(b) / max(n_blocks - 1, 1)
            coords[b, 1] = float(layer_id) / 100.0

        targets = block_means.reshape(-1, 1)
        best_params = None
        best_loss = float("inf")

        chunk_size = min(1024, n_blocks)

        for epoch in range(max_iter):
            for start in range(0, n_blocks, chunk_size):
                end = min(start + chunk_size, n_blocks)
                c = coords[start:end]
                t = targets[start:end]
                nc = end - start

                if use_siren:
                    pred = mlp.forward_siren(c)
                else:
                    pred = mlp.forward(c)

                loss = np.mean((pred - t) ** 2) * (nc / n_blocks)
                if loss < best_loss and start == 0 and epoch == 0:
                    best_loss = loss
                    best_params = mlp.get_params()

                grad = 2.0 * (pred - t) / nc
                if use_siren:
                    h = np.sin(30.0 * (c @ mlp.w1 + mlp.b1))
                    grad_out = grad
                    grad_w_out = h.T @ grad_out
                    grad_b_out = np.sum(grad_out, axis=0)
                    dh = (
                        grad_out
                        @ mlp.w_out.T
                        * (30.0 * np.cos(30.0 * (c @ mlp.w1 + mlp.b1)))
                    )
                    grad_w1 = c.T @ dh
                    grad_b1 = np.sum(dh, axis=0)
                else:
                    h1 = _silu(c @ mlp.w1 + mlp.b1)
                    grad_out = grad
                    grad_w_out = h1.T @ grad_out
                    grad_b_out = np.sum(grad_out, axis=0)
                    dh2 = grad_out @ mlp.w_out.T
                    dsilu = h1 * (1.0 - h1 * h1 / (1.0 + np.exp(-h1)))
                    dh1 = dh2 * dsilu
                    grad_w1 = c.T @ dh1
                    grad_b1 = np.sum(dh1, axis=0)

                mlp.w1 -= lr * grad_w1
                mlp.b1 -= lr * grad_b1
                mlp.w_out -= lr * grad_w_out
                mlp.b_out -= lr * grad_b_out

            # Track global best loss for full data
            if use_siren:
                pred_full = mlp.forward_siren(coords)
            else:
                pred_full = mlp.forward(coords)
            full_loss = np.mean((pred_full - targets) ** 2)
            if full_loss < best_loss:
                best_loss = full_loss
                best_params = mlp.get_params()

            if epoch % 100 == 0 and epoch > 0:
                lr *= 0.5

        if best_params is not None:
            mlp.set_params(best_params)

        params_bytes = mlp.get_params()
        header = struct.pack("<III", n, n_blocks, hidden_dim)
        header += struct.pack("<f", block_means.mean())
        header += struct.pack("<f", block_means.std())
        header += struct.pack("<I", max_iter)
        data = bytes(header) + params_bytes

        compressed_size = len(data)
        meta = {
            "shape": orig_shape,
            "n": n,
            "n_blocks": n_blocks,
            "hidden_dim": hidden_dim,
            "block_size": block_size,
            "block_mean": float(block_means.mean()),
            "block_std": float(block_means.std()),
            "final_loss": float(best_loss),
            "layer_id": layer_id,
            "use_siren": use_siren,
        }

        if compressed_size < tensor.nbytes:
            return data, meta
        from . import BlockINT8Wrapper

        return BlockINT8Wrapper.compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if (
            metadata.get("_fallback", False)
            or "block_size" in metadata
            and metadata.get("block_mean", 0) == 0
            and metadata.get("n", 0) == 0
        ):
            from . import BlockINT8Wrapper

            return BlockINT8Wrapper.decompress(data, metadata)

        n, n_blocks, hidden_dim = struct.unpack_from("<III", data, 0)
        if hidden_dim > 4096 or hidden_dim <= 0:
            return BlockINT8Wrapper.decompress(data, metadata)
        pos = 12
        block_mean = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        block_std = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        max_iter = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        params_bytes = data[pos:]
        mlp = HypernetworkMLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1)
        mlp.set_params(params_bytes)

        use_siren = metadata.get("use_siren", True)
        layer_id = metadata.get("layer_id", 0)
        block_size = metadata.get("block_size", 64)

        coords = np.zeros((n_blocks, 2), dtype=np.float32)
        for b in range(n_blocks):
            coords[b, 0] = float(b) / max(n_blocks - 1, 1)
            coords[b, 1] = float(layer_id) / 100.0

        if use_siren:
            pred = mlp.forward_siren(coords)
        else:
            pred = mlp.forward(coords)

        block_means = pred.ravel() * block_std + block_mean
        result = np.zeros(n, dtype=np.float32)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            result[start:end] = block_means[b]

        shape = metadata.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result


class BlockwiseINRCompression:
    """Block-wise Implicit Neural Representation compression.

    Splits tensor into blocks, trains a small MLP per block to map
    pixel position to value. Much more accurate than global MLP.
    """

    name = "blockwise_inr"
    category = "novel"

    def __init__(
        self,
        block_size: int = 32,
        hidden_dim: int = 16,
        max_iter: int = 200,
    ):
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter

    def compress(
        self, tensor: np.ndarray, layer_id: int = 0, **kwargs
    ) -> Tuple[bytes, dict]:
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)

        hd = kwargs.get("hidden_dim", self.hidden_dim)
        bs = kwargs.get("block_size", self.block_size)
        mi = kwargs.get("max_iter", self.max_iter)
        n_blocks = max(1, int(math.ceil(n / bs)))

        rng = np.random.RandomState(42)
        w1 = rng.randn(2, hd).astype(np.float32) * math.sqrt(2.0 / 2)
        b1 = np.zeros(hd, dtype=np.float32)
        w2 = rng.randn(hd, 1).astype(np.float32) * math.sqrt(2.0 / hd)
        b2 = np.zeros(1, dtype=np.float32)

        coords = np.zeros((n, 2), dtype=np.float32)
        pos_in_block = np.arange(n) % bs
        block_idx = np.arange(n) // bs
        coords[:, 0] = pos_in_block.astype(np.float32) / max(bs - 1, 1)
        coords[:, 1] = block_idx.astype(np.float32) / max(n_blocks - 1, 1)

        targets = flat.reshape(-1, 1)
        lr = 0.01
        best_w1, best_b1, best_w2, best_b2 = None, None, None, None
        best_loss = float("inf")
        chunk_size = min(1024, n)

        for epoch in range(mi):
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                c = coords[start:end]
                t = targets[start:end]
                nc = end - start

                h = np.sin(30.0 * (c @ w1 + b1))
                pred = h @ w2 + b2

                grad = 2.0 * (pred - t) / nc
                grad_w2 = h.T @ grad
                grad_b2 = np.sum(grad, axis=0)
                dh = grad @ w2.T * (30.0 * np.cos(30.0 * (c @ w1 + b1)))
                grad_w1 = c.T @ dh
                grad_b1 = np.sum(dh, axis=0)

                w1 -= lr * grad_w1
                b1 -= lr * grad_b1
                w2 -= lr * grad_w2
                b2 -= lr * grad_b2

            # Full pass for loss tracking
            h_full = np.sin(30.0 * (coords @ w1 + b1))
            pred_full = h_full @ w2 + b2
            loss = np.mean((pred_full - targets) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_w1 = w1.copy()
                best_b1 = b1.copy()
                best_w2 = w2.copy()
                best_b2 = b2.copy()

            if epoch % 50 == 0 and epoch > 0:
                lr *= 0.5

        w1 = best_w1 if best_w1 is not None else w1
        b1 = best_b1 if best_b1 is not None else b1
        w2 = best_w2 if best_w2 is not None else w2
        b2 = best_b2 if best_b2 is not None else b2

        header = struct.pack("<III", n, hd, bs)
        buf = header
        for arr in [w1, b1, w2, b2]:
            buf += arr.astype(np.float32).tobytes()

        return bytes(buf), {
            "shape": orig_shape,
            "n": n,
            "hidden_dim": hd,
            "block_size": bs,
            "final_loss": float(best_loss),
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n, hd, bs = struct.unpack_from("<III", data, 0)
        pos = 12
        w1 = np.frombuffer(data[pos : pos + 2 * hd * 4], dtype=np.float32).reshape(
            2, hd
        )
        pos += 2 * hd * 4
        b1 = np.frombuffer(data[pos : pos + hd * 4], dtype=np.float32)
        pos += hd * 4
        w2 = np.frombuffer(data[pos : pos + hd * 4], dtype=np.float32).reshape(hd, 1)
        pos += hd * 4
        b2 = np.frombuffer(data[pos : pos + 4], dtype=np.float32).reshape(1)

        n_blocks = max(1, int(math.ceil(n / bs)))
        coords = np.zeros((n, 2), dtype=np.float32)
        pos_in_block = np.arange(n) % bs
        block_idx = np.arange(n) // bs
        coords[:, 0] = pos_in_block.astype(np.float32) / max(bs - 1, 1)
        coords[:, 1] = block_idx.astype(np.float32) / max(n_blocks - 1, 1)

        h = np.sin(30.0 * (coords @ w1 + b1))
        result = (h @ w2 + b2).ravel()

        shape = metadata.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result


class SimpleHypernetworkCompression:
    """Compress weight matrix via tiny MLP hypernetwork.

    Trains a 2-layer MLP: input(3) -> hidden(32) -> output(1)
    Input features: (normalized_layer, normalized_row, normalized_col)
    Stores only the ~1.5KB of MLP weights instead of the full matrix.
    """

    name = "simple_hypernetwork"
    category = "novel"

    def __init__(
        self,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 50,
        block_size: int = 32,
    ):
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.block_size = block_size

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

    def _train_mlp(self, coords: np.ndarray, targets: np.ndarray) -> Dict:
        n, d = coords.shape
        h = self.hidden_dim
        chunk_size = min(1024, n)

        W1 = np.random.randn(d, h).astype(np.float32) * np.sqrt(2.0 / d)
        b1 = np.zeros(h, dtype=np.float32)
        W2 = np.random.randn(h, 1).astype(np.float32) * np.sqrt(1.0 / h)
        b2 = np.zeros(1, dtype=np.float32)

        targets = targets.reshape(-1, 1).astype(np.float32)

        for epoch in range(self.epochs):
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                c = coords[start:end]
                t = targets[start:end]
                nc = end - start

                z1 = c @ W1 + b1
                a1 = self._relu(z1)
                z2 = a1 @ W2 + b2
                pred = z2

                error = pred - t

                dz2 = error / nc
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                dz1 = (dz2 @ W2.T) * (z1 > 0).astype(np.float32)
                dW1 = c.T @ dz1
                db1 = dz1.sum(axis=0)

                W2 -= self.lr * dW2
                b2 -= self.lr * db2.reshape(b2.shape)
                W1 -= self.lr * dW1
                b1 -= self.lr * db1

        return {
            "W1": W1.astype(np.float16),
            "b1": b1.astype(np.float16),
            "W2": W2.astype(np.float16),
            "b2": b2.astype(np.float16),
            "input_dim": d,
            "hidden_dim": h,
        }

    def _forward_mlp(self, coords: np.ndarray, weights: Dict) -> np.ndarray:
        W1 = weights["W1"].astype(np.float32)
        b1 = weights["b1"].astype(np.float32)
        W2 = weights["W2"].astype(np.float32)
        b2 = weights["b2"].astype(np.float32)

        z1 = coords @ W1 + b1
        a1 = self._relu(z1)
        z2 = a1 @ W2 + b2
        return z2.flatten()

    def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, Dict]:
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        n = len(flat_tensor)

        rows, cols = np.mgrid[0 : original_shape[0], 0 : original_shape[1]]
        coords = np.column_stack(
            [
                rows.flatten() / max(original_shape[0], 1),
                cols.flatten() / max(original_shape[1], 1),
            ]
        )

        mlp_weights = self._train_mlp(coords, flat_tensor)

        recon = self._forward_mlp(coords, mlp_weights)
        residual = flat_tensor - recon

        k = max(1, int(n * 0.05))
        top_k_idx = np.argpartition(-np.abs(residual), k)[:k]

        return residual[top_k_idx].astype(np.float16).tobytes(), {
            "mlp_weights": mlp_weights,
            "shape": original_shape,
            "residual_indices": top_k_idx.astype(np.int32),
            "dtype": str(tensor.dtype),
            "method": "simple_hypernetwork",
        }

    def decompress(self, data: bytes, meta: Dict) -> np.ndarray:
        shape = meta["shape"]
        mlp_weights = meta["mlp_weights"]

        rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
        coords = np.column_stack(
            [
                rows.flatten() / max(shape[0], 1),
                cols.flatten() / max(shape[1], 1),
            ]
        )

        recon = self._forward_mlp(coords, mlp_weights)

        if data:
            residual_vals = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            residual_idx = meta["residual_indices"]
            recon[residual_idx] += residual_vals

        return recon.reshape(shape).astype(np.float32)


class FourierFeatureCompression:
    """Use random Fourier features + linear regression for weight encoding.

    Maps coordinates through a fixed random Fourier feature map,
    then learns a linear mapping to weight values.
    Much simpler and faster than MLP, often comparable quality.
    """

    name = "fourier_feature_compress"
    category = "novel"

    def __init__(self, num_features: int = 256, scale: float = 1.0):
        self.num_features = num_features
        self.scale = scale

    def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, Dict]:
        shape = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        n = len(flat)

        rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
        coords = (
            np.column_stack(
                [
                    rows.flatten() / max(shape[0], 1),
                    cols.flatten() / max(shape[1], 1),
                ]
            )
            * self.scale
        )

        rng = np.random.RandomState(42)
        W = rng.randn(2, self.num_features).astype(np.float32) * self.scale
        b = rng.rand(self.num_features).astype(np.float32) * 2 * np.pi

        features = np.cos(coords @ W + b)

        lam = 0.01
        X = features
        XtX = X.T @ X + lam * np.eye(self.num_features, dtype=np.float32)
        Xty = X.T @ flat
        theta = np.linalg.solve(XtX, Xty)

        recon = X @ theta

        residual = flat - recon
        k = max(1, int(n * 0.02))
        top_k = np.argpartition(-np.abs(residual), k)[:k]

        return residual[top_k].astype(np.float16).tobytes(), {
            "theta": theta.astype(np.float16),
            "shape": shape,
            "residual_indices": top_k.astype(np.int32),
            "method": "fourier_feature",
        }

    def decompress(self, data: bytes, meta: Dict) -> np.ndarray:
        shape = meta["shape"]
        theta = meta["theta"].astype(np.float32)

        rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
        coords = (
            np.column_stack(
                [
                    rows.flatten() / max(shape[0], 1),
                    cols.flatten() / max(shape[1], 1),
                ]
            )
            * self.scale
        )

        rng = np.random.RandomState(42)
        W = rng.randn(2, self.num_features).astype(np.float32) * self.scale
        b = rng.rand(self.num_features).astype(np.float32) * 2 * np.pi
        features = np.cos(coords @ W + b)

        recon = features @ theta

        if data:
            residual_vals = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            recon[meta["residual_indices"]] += residual_vals

        return recon.reshape(shape).astype(np.float32)

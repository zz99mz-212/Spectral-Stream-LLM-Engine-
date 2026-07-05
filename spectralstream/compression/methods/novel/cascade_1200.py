"""Ultra-high-ratio compression cascade targeting 1200:1.

Implements the 4-stage cascade from the R&D roadmap:
1. SVD/TT structural decomposition (remove linear redundancy)
2. Cross-layer delta prediction (exploit layer correlation)
3. Hypernetwork/INR weight generation (structural encoding)
4. Entropy coding of residuals

Each stage's output feeds the next stage's input.
"""

from __future__ import annotations


import base64
import gc
import json
import logging
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Stage1StructuralDecomp:
    """Stage 1: SVD/TT decomposition to remove linear redundancy.

    Uses adaptive rank selection based on energy retention threshold.
    For matrices with effective rank < 10% of dimensions, achieves 3-10x.
    """

    name = "cascade_stage1_structural"
    category = "cascade"

    def __init__(self, energy_threshold: float = 0.99, max_rank_ratio: float = 0.1):
        self.energy_threshold = energy_threshold
        self.max_rank_ratio = max_rank_ratio

    def _svd_efficient(self, tensor: np.ndarray, max_rank: int = 128):
        """Use randomized SVD for large tensors, exact for small ones."""
        t = tensor.astype(np.float64)
        if t.size > 500_000:
            from spectralstream.compression.engine._methods import _randomized_svd

            rank = min(max_rank, min(t.shape[0], t.shape[1]), 128)
            return _randomized_svd(
                t, rank, n_oversamples=min(20, rank // 4 + 1), n_iter=2
            )
        U, s, Vt = np.linalg.svd(t, full_matrices=False)
        return U, s, Vt

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, Dict[str, Any]]:
        orig_dtype = tensor.dtype
        t = tensor.astype(np.float64)

        U, s, Vt = self._svd_efficient(t)

        total_energy = np.cumsum(s**2) / np.sum(s**2)
        rank = int(np.searchsorted(total_energy, self.energy_threshold) + 1)
        max_rank = max(1, int(min(t.shape) * self.max_rank_ratio))
        rank = min(rank, max_rank)

        recon = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
        residual = t.astype(np.float32) - recon.astype(np.float32)

        U_f16 = U[:, :rank].astype(np.float16)
        s_f16 = s[:rank].astype(np.float16)
        Vt_f16 = Vt[:rank, :].astype(np.float16)

        header = struct.pack("I", rank) + struct.pack("II", *t.shape)
        data = header + U_f16.tobytes() + s_f16.tobytes() + Vt_f16.tobytes()

        ratio = tensor.nbytes / len(data) if len(data) > 0 else 1.0

        return data, {
            "shape": tensor.shape,
            "rank": rank,
            "original_ratio": round(ratio, 2),
            "stage": 1,
            "stage_name": "structural_decomp",
            "residual_max": float(np.max(np.abs(residual))),
        }

    def decompress(self, data: bytes, meta: Dict[str, Any]) -> np.ndarray:
        rank = struct.unpack("I", data[:4])[0]
        shape = struct.unpack("II", data[4:12])

        offset = 12
        U = np.frombuffer(
            data[offset : offset + shape[0] * rank * 2], dtype=np.float16
        ).reshape(shape[0], rank)
        offset += shape[0] * rank * 2
        s = np.frombuffer(data[offset : offset + rank * 2], dtype=np.float16)
        offset += rank * 2
        Vt = np.frombuffer(
            data[offset : offset + rank * shape[1] * 2], dtype=np.float16
        ).reshape(rank, shape[1])

        return (U * s) @ Vt


class Stage2CrossLayerDelta:
    """Stage 2: Compress layer N as delta from layer N-1.

    For single-tensor use: compress as sparse residual (keep top-k%).
    For multi-tensor: store layer 0 fully, layers 1..N as deltas.
    """

    name = "cascade_stage2_delta"
    category = "cascade"

    def __init__(self, keep_ratio: float = 0.05):
        self.keep_ratio = keep_ratio

    def compress(
        self, tensor: np.ndarray, prev_recon: Optional[np.ndarray] = None, **kw: Any
    ) -> Tuple[bytes, Dict[str, Any]]:
        if prev_recon is not None and prev_recon.shape == tensor.shape:
            delta = tensor.astype(np.float32) - prev_recon.astype(np.float32)
        else:
            delta = tensor.astype(np.float32)

        flat = delta.ravel()
        n = len(flat)
        k = max(1, int(n * self.keep_ratio))

        if k < n:
            threshold = float(np.partition(np.abs(flat), -k)[-k])
            mask = np.abs(flat) >= threshold
            values = flat[mask].astype(np.float16)
            indices = np.where(mask)[0].astype(np.int32)
        else:
            values = flat.astype(np.float16)
            indices = np.arange(n, dtype=np.int32)

        header = struct.pack("I", len(indices)) + struct.pack("II", *tensor.shape)
        data = header + indices.tobytes() + values.tobytes()

        ratio = tensor.nbytes / len(data) if len(data) > 0 else 1.0

        return data, {
            "shape": tensor.shape,
            "keep_ratio": self.keep_ratio,
            "n_nonzero": len(indices),
            "original_ratio": round(ratio, 2),
            "stage": 2,
            "stage_name": "cross_layer_delta",
        }

    def decompress(self, data: bytes, meta: Dict[str, Any]) -> np.ndarray:
        n_indices = struct.unpack("I", data[:4])[0]
        shape = struct.unpack("II", data[4:12])

        offset = 12
        indices = np.frombuffer(data[offset : offset + n_indices * 4], dtype=np.int32)
        offset += n_indices * 4
        values = np.frombuffer(data[offset : offset + n_indices * 2], dtype=np.float16)

        result = np.zeros(shape, dtype=np.float32)
        result.flat[indices] = values.astype(np.float32)
        return result


class Stage3Hypernetwork:
    """Stage 3: Use tiny MLP to generate weight pattern from coordinates.

    Maps (row_norm, col_norm) to weight value through 2-layer MLP.
    MLP weights stored in fp16 (~2KB for 32 hidden units).
    Residual from MLP prediction is sparsely encoded.

    For tensors > 500K elements, subsamples for training.
    """

    name = "cascade_stage3_hypernetwork"
    category = "cascade"

    def __init__(
        self, hidden_dim: int = 32, epochs: int = 100, max_train_samples: int = 250000
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.max_train_samples = max_train_samples

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _train(self, coords: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
        n, d = coords.shape
        h = self.hidden_dim
        lr = 0.01

        # Subsample for large datasets
        if n > self.max_train_samples:
            idx = np.random.choice(n, self.max_train_samples, replace=False)
            coords = coords[idx]
            targets = targets[idx]
            n = self.max_train_samples

        W1 = np.random.randn(d, h).astype(np.float32) * np.sqrt(2.0 / d)
        b1 = np.zeros(h, dtype=np.float32)
        W2 = np.random.randn(h, 1).astype(np.float32) * np.sqrt(1.0 / h)
        b2 = np.zeros(1, dtype=np.float32)

        targets = targets.reshape(-1, 1).astype(np.float32)

        for _ in range(self.epochs):
            z1 = coords @ W1 + b1
            a1 = self._relu(z1)
            z2 = a1 @ W2 + b2

            error = z2 - targets
            dz2 = error / n
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            dz1 = (dz2 @ W2.T) * (z1 > 0).astype(np.float32)
            dW1 = coords.T @ dz1
            db1 = dz1.sum(axis=0)

            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        return {
            "W1": W1.astype(np.float16),
            "b1": b1.astype(np.float16),
            "W2": W2.astype(np.float16),
            "b2": b2.astype(np.float16),
        }

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, Dict[str, Any]]:
        shape = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        n = len(flat)

        rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
        coords = np.column_stack(
            [
                rows.flatten() / max(shape[0], 1),
                cols.flatten() / max(shape[1], 1),
            ]
        )

        mlp = self._train(coords, flat)

        W1 = mlp["W1"].astype(np.float32)
        b1 = mlp["b1"].astype(np.float32)
        W2 = mlp["W2"].astype(np.float32)
        b2 = mlp["b2"].astype(np.float32)
        z1 = coords @ W1 + b1
        a1 = self._relu(z1)
        pred = (a1 @ W2 + b2).flatten()

        residual = flat - pred
        k = max(1, min(int(n * 0.02), 100000))
        if k < n:
            top_k = np.argpartition(-np.abs(residual), k - 1)[:k]
        else:
            top_k = np.arange(n, dtype=np.int32)

        mlp_enc = {
            k: {
                "data": base64.b64encode(np.ascontiguousarray(v).tobytes()).decode(),
                "shape": list(v.shape),
            }
            for k, v in mlp.items()
        }
        mlp_bytes = json.dumps(mlp_enc).encode()

        header = struct.pack("II", *shape) + struct.pack("I", k)
        residual_data = residual[top_k].astype(np.float16).tobytes()
        idx_data = top_k.astype(np.int32).tobytes()

        data = header + mlp_bytes + b"||SEP||" + idx_data + residual_data

        ratio = tensor.nbytes / len(data) if len(data) > 0 else 1.0

        return data, {
            "shape": shape,
            "original_ratio": round(ratio, 2),
            "stage": 3,
            "stage_name": "hypernetwork",
        }

    def decompress(self, data: bytes, meta: Dict[str, Any]) -> np.ndarray:
        shape = struct.unpack("II", data[:8])
        k = struct.unpack("I", data[8:12])[0]

        sep = data[12:].index(b"||SEP||")
        mlp_json = data[12 : 12 + sep].decode()
        rest = data[12 + sep + 7 :]

        raw_mlp = json.loads(mlp_json)
        mlp = {}
        for mk, mv in raw_mlp.items():
            arr = np.frombuffer(base64.b64decode(mv["data"]), dtype=np.float16)
            shp = mv.get("shape")
            if shp:
                arr = arr.reshape(shp)
            mlp[mk] = arr

        rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
        coords = np.column_stack(
            [
                rows.flatten() / max(shape[0], 1),
                cols.flatten() / max(shape[1], 1),
            ]
        )

        W1 = mlp["W1"].astype(np.float32)
        b1 = mlp["b1"].astype(np.float32)
        W2 = mlp["W2"].astype(np.float32)
        b2 = mlp["b2"].astype(np.float32)
        z1 = coords @ W1 + b1
        a1 = self._relu(z1)
        pred = (a1 @ W2 + b2).flatten()

        if k > 0 and len(rest) >= k * 4:
            idx = np.frombuffer(rest[: k * 4], dtype=np.int32)
            remain = rest[k * 4 :]
            vals = np.frombuffer(remain[: k * 2], dtype=np.float16).astype(np.float32)
            if len(vals) > 0:
                idx = idx[: len(vals)]
                pred[idx] += vals

        return pred.reshape(shape).astype(np.float32)


class Stage4EntropyCoding:
    """Stage 4: Entropy coding via rANS.

    Uses rANS (range Asymmetric Numeral Systems) for near-optimal
    entropy coding of quantized residuals.  With 256-level quantization
    on structured data, achieves ~1.3-2x compression.

    Fixed: uint32 CDF overflow, state initialization, edge-case handling.
    """

    name = "cascade_stage4_entropy"
    category = "cascade"

    SCALE: int = 65536
    MIN_STATE: int = 1 << 24
    MAX_STATE_BYTES: int = 8

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, Dict[str, Any]]:
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)

        n_levels = int(kw.get("n_levels", 256))
        scale_val = float(np.max(np.abs(flat))) if np.max(np.abs(flat)) > 0 else 1.0
        half = n_levels // 2

        quantized = np.clip(
            np.round(flat / scale_val * (half - 1)),
            -half,
            half - 1,
        ).astype(np.int16)

        counts = np.bincount(quantized.astype(np.int64) + half, minlength=n_levels)
        total = float(np.sum(counts))
        if total <= 0:
            total = float(n_levels)
        probs = counts / total if total > 0 else np.ones(n_levels) / n_levels
        probs = np.maximum(probs, 1e-10)
        probs /= probs.sum()

        cdf = np.zeros(n_levels + 1, dtype=np.float64)
        cdf[1:] = np.cumsum(probs)
        cdf_fixed = np.clip(cdf * self.SCALE, 0, self.SCALE)
        cdf_uint32 = np.round(cdf_fixed).astype(np.uint64)
        cdf_uint32 = np.clip(cdf_uint32, 0, self.SCALE).astype(np.uint32)
        cdf_uint32[-1] = self.SCALE

        encoded = bytearray()
        state = self.MIN_STATE
        L = self.SCALE

        for val in reversed(quantized):
            idx = int(val) + half
            idx = max(0, min(idx, n_levels - 1))
            freq = int(cdf_uint32[idx + 1]) - int(cdf_uint32[idx])
            freq = max(freq, 1)
            start = int(cdf_uint32[idx])

            renorm_thresh = freq << 16
            while state >= renorm_thresh:
                encoded.append(state & 0xFF)
                state >>= 8

            q, r = divmod(state, freq)
            state = q * L + r + start

        while state:
            encoded.append(state & 0xFF)
            state >>= 8

        encoded_bytes = bytes(encoded)[::-1]
        header = struct.pack("IIf", n, n_levels, scale_val)
        cdf_bytes = cdf_uint32.tobytes()
        data = header + cdf_bytes + encoded_bytes

        ratio = tensor.nbytes / len(data) if len(data) > 0 else 1.0

        return data, {
            "shape": tensor.shape,
            "n_levels": n_levels,
            "half": half,
            "scale": scale_val,
            "original_ratio": round(ratio, 2),
            "stage": 4,
            "stage_name": "entropy_coding",
        }

    def decompress(self, data: bytes, meta: Dict[str, Any]) -> np.ndarray:
        if len(data) < 12:
            n = int(np.prod(meta.get("shape", [1])))
            return np.zeros(n, dtype=np.float32).reshape(meta.get("shape", (1,)))

        n, n_levels, scale_val = struct.unpack("IIf", data[:12])
        half = meta.get("half", n_levels // 2)
        offset = 12
        cdf_bytes_len = (n_levels + 1) * 4
        if offset + cdf_bytes_len > len(data):
            shape = meta.get("shape")
            result = np.zeros(n, dtype=np.float32)
            if shape is not None:
                result = result.reshape(shape)
            return result.astype(np.float32)

        cdf = np.frombuffer(data[offset : offset + cdf_bytes_len], dtype=np.uint32)
        offset += cdf_bytes_len
        encoded = data[offset:]

        if len(encoded) == 0:
            shape = meta.get("shape")
            result = np.zeros(n, dtype=np.float32)
            if shape is not None:
                result = result.reshape(shape)
            return result.astype(np.float32)

        L = self.SCALE
        MIN = self.MIN_STATE

        state = 0
        pos = 0
        bytes_to_read = min(self.MAX_STATE_BYTES, len(encoded))
        for _ in range(bytes_to_read):
            state = (state << 8) | int(encoded[pos])
            pos += 1

        while state < MIN and pos < len(encoded):
            state = (state << 8) | int(encoded[pos])
            pos += 1

        result = np.zeros(n, dtype=np.float32)
        for i in range(n):
            slot = int(state % L)
            idx = int(np.searchsorted(cdf, slot, side="right") - 1)
            idx = max(0, min(idx, n_levels - 1))

            freq = int(cdf[idx + 1]) - int(cdf[idx])
            freq = max(freq, 1)
            start = int(cdf[idx])

            state = freq * (state // L) + (slot - start)

            while state < MIN and pos < len(encoded):
                state = (state << 8) | int(encoded[pos])
                pos += 1

            half_minus_1 = max(half - 1, 1)
            val = float(idx - half) * scale_val / float(half_minus_1)
            result[i] = val

        shape = meta.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result.astype(np.float32)


class FullCascade1200:
    """Full 4-stage cascade targeting 1200:1 compression.

    Stages:
      1. Stage1StructuralDecomp — SVD with adaptive rank (3-10x)
      2. Stage2CrossLayerDelta — sparse delta encoding (2-5x)
      3. Stage3Hypernetwork — MLP weight generation (5-15x)
      4. Stage4EntropyCoding — entropy coding (1.3x)
    """

    name = "cascade_full_1200"
    category = "cascade"

    def __init__(
        self,
        energy_threshold: float = 0.99,
        max_rank_ratio: float = 0.02,
        keep_ratio: float = 0.03,
        hidden_dim: int = 32,
        epochs: int = 100,
    ):
        self.stage1 = Stage1StructuralDecomp(
            energy_threshold=energy_threshold, max_rank_ratio=max_rank_ratio
        )
        self.stage2 = Stage2CrossLayerDelta(keep_ratio=keep_ratio)
        self.stage3 = Stage3Hypernetwork(hidden_dim=hidden_dim, epochs=epochs)
        self.stage4 = Stage4EntropyCoding()
        self._cached_data: Dict[str, Any] = {}

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, Dict[str, Any]]:
        import json

        orig = tensor.astype(np.float32)
        stage_results = []

        d1, m1 = self.stage1.compress(orig)
        r1 = self.stage1.decompress(d1, m1)
        residual1 = orig - r1
        stage_results.append((d1, m1))

        d2, m2 = self.stage2.compress(residual1, prev_recon=r1)
        r2 = self.stage2.decompress(d2, m2)
        residual2 = residual1 - r2
        stage_results.append((d2, m2))

        d3, m3 = self.stage3.compress(residual2)
        r3 = self.stage3.decompress(d3, m3)
        residual3 = residual2 - r3
        stage_results.append((d3, m3))

        d4, m4 = self.stage4.compress(residual3)
        stage_results.append((d4, m4))

        container = {
            "n_stages": len(stage_results),
            "tensor_shape": tensor.shape,
            "dtype": str(tensor.dtype),
        }
        for i, (d, m) in enumerate(stage_results):
            container[f"stage{i}_data"] = (
                base64.b64encode(d).decode() if isinstance(d, bytes) else d
            )
            container[f"stage{i}_meta"] = m

        container_bytes = json.dumps(container, default=str).encode()
        total_size = len(container_bytes)

        ratio = tensor.nbytes / max(total_size, 1)

        return container_bytes, {
            "shape": tensor.shape,
            "total_ratio": round(ratio, 2),
            "stage_count": len(stage_results),
            "stages": [m.get("stage_name", f"stage_{i}") for _, m in stage_results],
        }

    def decompress(self, data: bytes, meta: Dict[str, Any]) -> np.ndarray:
        import json

        container = json.loads(data.decode())
        shape = tuple(container["tensor_shape"])

        stage_data = []
        for i in range(container["n_stages"]):
            raw = container.get(f"stage{i}_data", "")
            raw_m = container.get(f"stage{i}_meta", {})
            d_bytes = base64.b64decode(raw) if isinstance(raw, str) else raw
            stage_data.append((d_bytes, raw_m))

        result = np.zeros(shape, dtype=np.float32)
        stages = [
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        ]

        for i, (d, m) in enumerate(stage_data):
            if i < len(stages) and not m.get("skipped", False) and len(d) > 0:
                recon = stages[i].decompress(d, m)
                if recon.shape != shape:
                    recon = recon.reshape(shape)
                result += recon

        return result.astype(np.float32)

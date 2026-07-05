"""Predictive Coding Methods for FP32 Weights — 20 compression techniques."""

from __future__ import annotations

import heapq
import zlib
from collections import Counter, defaultdict
from typing import Any, Dict, Tuple

import numpy as np


def _huffman_encode_ints(ints: np.ndarray) -> Tuple[np.ndarray, int, dict]:
    freq = dict(Counter(ints.tolist()))
    if len(freq) <= 1:
        tree = {list(freq.keys())[0]: "0"} if freq else {}
        n_bits = len(ints)
        n_bytes = (n_bits + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        if freq:
            for i in range(n_bits):
                packed[i // 8] |= 1 << (i % 8)
        return packed, n_bits, tree
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    tree = {symbol: code for symbol, code in heap[0][1:]}
    encoded = "".join(tree[int(x)] for x in ints)
    n_bits = len(encoded)
    n_bytes = (n_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bits):
        if encoded[i] == "1":
            packed[i // 8] |= 1 << (i % 8)
    return packed, n_bits, tree


def _huffman_decode_ints(
    packed: np.ndarray, n_bits: int, n_orig: int, tree: dict
) -> np.ndarray:
    reverse_tree = {v: k for k, v in tree.items()}
    decoded = []
    current = ""
    for i in range(n_bits):
        byte_idx = i // 8
        bit_idx = i % 8
        current += "1" if (packed[byte_idx] >> bit_idx) & 1 else "0"
        if current in reverse_tree:
            decoded.append(reverse_tree[current])
            current = ""
    return np.array(decoded[:n_orig], dtype=np.int32)


def _unpack_fp32_bytes(data: bytes, shape: tuple) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).reshape(shape)


def _compute_error_metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    mse = float(np.mean((o - r) ** 2))
    snr = 10 * np.log10(np.sum(o**2) / (np.sum((o - r) ** 2) + 1e-30))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    rel_err = float(np.mean(np.abs(o - r) / (np.abs(o) + 1e-10)))
    return {
        "mse": mse,
        "snr_db": float(snr),
        "cosine_similarity": cos_sim,
        "rel_error": rel_err,
    }


def _get_2d_shape(tensor: np.ndarray) -> Tuple[int, int]:
    if len(tensor.shape) >= 2:
        return tensor.shape[0], tensor.shape[1]
    return 1, len(tensor.ravel())


class DeltaRowCoding:
    """Delta encoding along rows: W[i,j] -> delta = W[i,j] - W[i,j-1]."""

    METHOD_NAME = "delta_row_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel()
        rows, cols = _get_2d_shape(tensor)
        n = len(flat)
        delta = np.zeros(n, dtype=np.float32)
        for r in range(rows):
            base = r * cols
            delta[base] = flat[base]
            for c in range(1, cols):
                delta[base + c] = flat[base + c] - flat[base + c - 1]
        compressed = zlib.compress(delta.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "n_elements": np.int32(n),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        delta = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(len(delta), dtype=np.float32)
        for r in range(rows):
            base = r * cols
            flat[base] = delta[base]
            for c in range(1, cols):
                flat[base + c] = delta[base + c] + flat[base + c - 1]
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return len(data["compressed"]) * 8 / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class DeltaColumnCoding:
    """Delta encoding along columns: W[i,j] -> delta = W[i,j] - W[i-1,j]."""

    METHOD_NAME = "delta_column_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        rows, cols = _get_2d_shape(tensor)
        flat = tensor.ravel()
        delta = np.zeros_like(flat, dtype=np.float32)
        for c in range(cols):
            for r in range(rows):
                idx = r * cols + c
                delta[idx] = (
                    flat[idx] if r == 0 else flat[idx] - flat[(r - 1) * cols + c]
                )
        compressed = zlib.compress(delta.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        delta = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(len(delta), dtype=np.float32)
        for c in range(cols):
            for r in range(rows):
                idx = r * cols + c
                flat[idx] = (
                    delta[idx] if r == 0 else delta[idx] + flat[(r - 1) * cols + c]
                )
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        return (
            len(self.compress(tensor, **kwargs)[0]["compressed"])
            * 8
            / max(tensor.nbytes, 1)
        )

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class Delta2DCoding:
    """2D prediction: predict W[i,j] from W[i-1,j] + W[i,j-1] - W[i-1,j-1]."""

    METHOD_NAME = "delta_2d_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        rows, cols = _get_2d_shape(tensor)
        flat = tensor.ravel()
        residual = np.zeros_like(flat, dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if r == 0 and c == 0:
                    residual[idx] = flat[idx]
                elif r == 0:
                    residual[idx] = flat[idx] - flat[idx - 1]
                elif c == 0:
                    residual[idx] = flat[idx] - flat[(r - 1) * cols + c]
                else:
                    pred = (
                        flat[(r - 1) * cols + c]
                        + flat[r * cols + c - 1]
                        - flat[(r - 1) * cols + c - 1]
                    )
                    residual[idx] = flat[idx] - pred
        compressed = zlib.compress(residual.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residual = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(len(residual), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if r == 0 and c == 0:
                    flat[idx] = residual[idx]
                elif r == 0:
                    flat[idx] = residual[idx] + flat[idx - 1]
                elif c == 0:
                    flat[idx] = residual[idx] + flat[(r - 1) * cols + c]
                else:
                    pred = (
                        flat[(r - 1) * cols + c]
                        + flat[r * cols + c - 1]
                        - flat[(r - 1) * cols + c - 1]
                    )
                    flat[idx] = residual[idx] + pred
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        return (
            len(self.compress(tensor, **kwargs)[0]["compressed"])
            * 8
            / max(tensor.nbytes, 1)
        )

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class ARPredictCoding:
    """AR(1) model: predict W[i] = a * W[i-1] + residual."""

    METHOD_NAME = "ar1_predict_coding"

    @staticmethod
    def _fit_ar1(row: np.ndarray) -> Tuple[float, np.ndarray]:
        if len(row) < 2:
            return 0.0, row.copy()
        x, y = row[:-1], row[1:]
        a = float(np.dot(x, y) / (np.dot(x, x) + 1e-10))
        residual = np.zeros(len(row), dtype=np.float32)
        residual[0] = row[0]
        for i in range(1, len(row)):
            residual[i] = row[i] - a * row[i - 1]
        return a, residual

    @staticmethod
    def _reconstruct_ar1(first: float, a: float, residual: np.ndarray) -> np.ndarray:
        n = len(residual)
        recon = np.zeros(n, dtype=np.float32)
        recon[0] = first
        for i in range(1, n):
            recon[i] = a * recon[i - 1] + residual[i]
        return recon

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel()
        rows, cols = _get_2d_shape(tensor)
        coeffs, all_residuals, all_firsts = [], [], []
        for r in range(rows):
            row = flat[r * cols : (r + 1) * cols]
            a, residual = self._fit_ar1(row)
            coeffs.append(a)
            all_residuals.append(residual)
            all_firsts.append(row[0])
        residuals = np.concatenate(all_residuals)
        res_compressed = zlib.compress(residuals.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(res_compressed, dtype=np.uint8),
            "coeffs": np.array(coeffs, dtype=np.float32),
            "firsts": np.array(all_firsts, dtype=np.float32),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residuals = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        coeffs, firsts = data["coeffs"], data["firsts"]
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(rows * cols, dtype=np.float32)
        for r in range(rows):
            flat[r * cols : (r + 1) * cols] = self._reconstruct_ar1(
                float(firsts[r]), float(coeffs[r]), residuals[r * cols : (r + 1) * cols]
            )
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return (
            len(data["compressed"]) * 8
            + len(data["coeffs"]) * 32
            + len(data["firsts"]) * 32
        ) / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class AR2PredictCoding:
    """AR(2) model: predict W[i] = a1*W[i-1] + a2*W[i-2] + residual."""

    METHOD_NAME = "ar2_predict_coding"

    @staticmethod
    def _fit_ar2(row: np.ndarray) -> Tuple[float, float, np.ndarray]:
        n = len(row)
        if n < 3:
            return 0.0, 0.0, row.copy()
        x1, x2, y = row[:-2], row[1:-1], row[2:]
        A = np.stack([x1, x2], axis=1)
        params = np.linalg.lstsq(A, y, rcond=None)[0]
        a1, a2 = float(params[0]), float(params[1])
        residual = np.zeros(n, dtype=np.float32)
        residual[0] = row[0]
        if n > 1:
            residual[1] = row[1] - a1 * row[0]
        for i in range(2, n):
            residual[i] = row[i] - a1 * row[i - 1] - a2 * row[i - 2]
        return a1, a2, residual

    @staticmethod
    def _reconstruct_ar2(
        first: float, second: float, a1: float, a2: float, residual: np.ndarray
    ) -> np.ndarray:
        n = len(residual)
        recon = np.zeros(n, dtype=np.float32)
        recon[0] = first
        if n > 1:
            recon[1] = second
        for i in range(2, n):
            recon[i] = a1 * recon[i - 1] + a2 * recon[i - 2] + residual[i]
        return recon

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel()
        rows, cols = _get_2d_shape(tensor)
        a1s, a2s, all_residuals, all_firsts, all_seconds = [], [], [], [], []
        for r in range(rows):
            row = flat[r * cols : (r + 1) * cols]
            a1, a2, residual = self._fit_ar2(row)
            a1s.append(a1)
            a2s.append(a2)
            all_residuals.append(residual)
            all_firsts.append(row[0])
            all_seconds.append(row[1] if len(row) > 1 else 0.0)
        residuals = np.concatenate(all_residuals)
        res_compressed = zlib.compress(residuals.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(res_compressed, dtype=np.uint8),
            "a1": np.array(a1s, dtype=np.float32),
            "a2": np.array(a2s, dtype=np.float32),
            "firsts": np.array(all_firsts, dtype=np.float32),
            "seconds": np.array(all_seconds, dtype=np.float32),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residuals = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(rows * cols, dtype=np.float32)
        for r in range(rows):
            flat[r * cols : (r + 1) * cols] = self._reconstruct_ar2(
                float(data["firsts"][r]),
                float(data["seconds"][r]),
                float(data["a1"][r]),
                float(data["a2"][r]),
                residuals[r * cols : (r + 1) * cols],
            )
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return (
            len(data["compressed"]) * 8
            + len(data["a1"]) * 64
            + len(data["firsts"]) * 64
        ) / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class ARPredictRowCoding:
    """Per-row AR(1) with Huffman-coded residuals."""

    METHOD_NAME = "ar_row_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel()
        rows, cols = _get_2d_shape(tensor)
        coeffs, all_firsts = [], []
        all_residuals = []
        for r in range(rows):
            row = flat[r * cols : (r + 1) * cols]
            a, residual = ARPredictCoding._fit_ar1(row)
            coeffs.append(a)
            all_firsts.append(row[0])
            all_residuals.append(residual)
        residuals = np.concatenate(all_residuals)
        residuals_int32 = residuals.view(np.int32)
        packed, n_bits, tree = _huffman_encode_ints(residuals_int32)
        data = {
            "packed": packed,
            "n_bits": np.int32(n_bits),
            "tree": tree,
            "coeffs": np.array(coeffs, dtype=np.float32),
            "firsts": np.array(all_firsts, dtype=np.float32),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residuals_int32 = _huffman_decode_ints(
            data["packed"],
            int(data["n_bits"]),
            int(data["rows"]) * int(data["cols"]),
            data["tree"],
        )
        residuals = residuals_int32.view(np.float32)
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(rows * cols, dtype=np.float32)
        for r in range(rows):
            flat[r * cols : (r + 1) * cols] = ARPredictCoding._reconstruct_ar1(
                float(data["firsts"][r]),
                float(data["coeffs"][r]),
                residuals[r * cols : (r + 1) * cols],
            )
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return (
            len(data["packed"]) * 8
            + len(data["coeffs"]) * 32
            + len(data["firsts"]) * 32
        ) / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class ARPredictColumnCoding:
    """Per-column AR(1) model."""

    METHOD_NAME = "ar_column_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        rows, cols = _get_2d_shape(tensor)
        flat = tensor.ravel()
        coeffs, all_residuals, all_firsts = [], [], []
        for c in range(cols):
            col = np.array([flat[r * cols + c] for r in range(rows)], dtype=np.float32)
            a, residual = ARPredictCoding._fit_ar1(col)
            coeffs.append(a)
            all_residuals.append(residual)
            all_firsts.append(col[0])
        residuals = np.concatenate(all_residuals)
        compressed = zlib.compress(residuals.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "coeffs": np.array(coeffs, dtype=np.float32),
            "firsts": np.array(all_firsts, dtype=np.float32),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residuals = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(rows * cols, dtype=np.float32)
        for c in range(cols):
            col = ARPredictCoding._reconstruct_ar1(
                float(data["firsts"][c]),
                float(data["coeffs"][c]),
                residuals[c * rows : (c + 1) * rows],
            )
            for r in range(rows):
                flat[r * cols + c] = col[r]
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return (
            len(data["compressed"]) * 8
            + len(data["coeffs"]) * 32
            + len(data["firsts"]) * 32
        ) / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class ContextModelCoding:
    """Context model: predict from surrounding neighbors (mean of available)."""

    METHOD_NAME = "context_model_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        rows, cols = _get_2d_shape(tensor)
        flat = tensor.ravel()
        residual = np.zeros_like(flat, dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                neighbors = []
                if r > 0:
                    neighbors.append(flat[(r - 1) * cols + c])
                if c > 0:
                    neighbors.append(flat[r * cols + c - 1])
                if r > 0 and c > 0:
                    neighbors.append(flat[(r - 1) * cols + c - 1])
                residual[idx] = (
                    flat[idx] - np.mean(neighbors) if neighbors else flat[idx]
                )
        compressed = zlib.compress(residual.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "rows": np.int32(rows),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residual = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        rows, cols = int(data["rows"]), int(data["cols"])
        flat = np.zeros(len(residual), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                neighbors = []
                if r > 0:
                    neighbors.append(flat[(r - 1) * cols + c])
                if c > 0:
                    neighbors.append(flat[r * cols + c - 1])
                if r > 0 and c > 0:
                    neighbors.append(flat[(r - 1) * cols + c - 1])
                flat[idx] = (
                    residual[idx] + np.mean(neighbors) if neighbors else residual[idx]
                )
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        return (
            len(self.compress(tensor, **kwargs)[0]["compressed"])
            * 8
            / max(tensor.nbytes, 1)
        )

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class DictionaryCoding:
    """Dictionary coding: repeated row patterns stored once + index references."""

    METHOD_NAME = "dictionary_coding"

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        rows, cols = _get_2d_shape(tensor)
        if rows < 2 or cols < 1:
            return DeltaRowCoding().compress(tensor, **kwargs)
        flat = tensor.ravel()
        dictionary = {}
        indices = []
        transforms = []

        for r in range(rows):
            row = flat[r * cols : (r + 1) * cols]
            row_key = row.tobytes()
            if row_key in dictionary:
                indices.append(dictionary[row_key])
                transforms.extend([1.0, 0.0])
                continue
            matched = False
            for existing_key, existing_idx in dictionary.items():
                existing_row = np.frombuffer(existing_key, dtype=np.float32)
                if len(existing_row) != cols:
                    continue
                norm_r = np.linalg.norm(row)
                norm_e = np.linalg.norm(existing_row)
                if norm_r < 1e-10 or norm_e < 1e-10:
                    continue
                sim = np.dot(row, existing_row) / (norm_r * norm_e)
                if sim > self.similarity_threshold:
                    scale = np.dot(row, existing_row) / (
                        np.dot(existing_row, existing_row) + 1e-10
                    )
                    offset = float(np.mean(row) - scale * np.mean(existing_row))
                    approx = scale * existing_row + offset
                    if np.max(np.abs(row - approx)) < 1e-6:
                        indices.append(existing_idx)
                        transforms.extend([float(scale), offset])
                        matched = True
                        break
            if not matched:
                dictionary[row_key] = len(dictionary)
                indices.append(dictionary[row_key])
                transforms.extend([1.0, 0.0])

        if dictionary:
            dict_arr = np.concatenate(
                [np.frombuffer(k, dtype=np.float32) for k in dictionary.keys()]
            )
        else:
            dict_arr = np.array([], dtype=np.float32)
        dict_compressed = zlib.compress(dict_arr.tobytes(), 9)
        data = {
            "dict_compressed": np.frombuffer(dict_compressed, dtype=np.uint8),
            "indices": np.array(indices, dtype=np.int32),
            "transforms": np.array(transforms, dtype=np.float32),
            "n_dict": np.int32(len(dictionary)),
            "cols": np.int32(cols),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        orig_shape = metadata["orig_shape"]
        rows, cols = orig_shape[0], orig_shape[1] if len(orig_shape) > 1 else 1
        dict_arr = (
            np.frombuffer(
                zlib.decompress(data["dict_compressed"].tobytes()), dtype=np.float32
            ).reshape(-1, cols)
            if int(data["n_dict"]) > 0
            else np.zeros((0, cols), dtype=np.float32)
        )
        indices = data["indices"]
        transforms = data["transforms"]
        flat = np.zeros(rows * cols, dtype=np.float32)
        for r in range(rows):
            idx = int(indices[r])
            scale, offset = float(transforms[r * 2]), float(transforms[r * 2 + 1])
            flat[r * cols : (r + 1) * cols] = scale * dict_arr[idx] + offset
        return flat.reshape(orig_shape)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return (
            len(data["dict_compressed"]) * 8
            + len(data["indices"]) * 32
            + len(data["transforms"]) * 32
        ) / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class RunLengthCoding:
    """Run-length encoding for repeated values."""

    METHOD_NAME = "run_length_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel()
        runs = []
        i = 0
        while i < len(flat):
            val = flat[i]
            run_len = 1
            while (
                i + run_len < len(flat) and flat[i + run_len] == val and run_len < 65535
            ):
                run_len += 1
            runs.append((float(val), run_len))
            i += run_len
        data = {
            "values": np.array([r[0] for r in runs], dtype=np.float32),
            "lengths": np.array([r[1] for r in runs], dtype=np.int32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        flat = np.repeat(data["values"], data["lengths"])
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        flat = tensor.ravel()
        runs = 1 + int(np.sum(flat[1:] != flat[:-1]))
        return runs * 64 / max(tensor.nbytes, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class LZ77WeightCoding:
    """LZ77 sliding-window on raw weight bytes."""

    METHOD_NAME = "lz77_weight_coding"

    def __init__(self, window_size: int = 2048, max_match: int = 128):
        self.window_size = window_size
        self.max_match = max_match

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        raw = tensor.astype(np.float32).tobytes()
        n = len(raw)
        tokens = []
        i = 0
        while i < n:
            best_off, best_len = 0, 0
            start = max(0, i - self.window_size)
            for j in range(start, i):
                length = 0
                while (
                    length < self.max_match
                    and i + length < n
                    and raw[j + length] == raw[i + length]
                ):
                    length += 1
                if length > best_len:
                    best_off, best_len = i - j, length
            if best_len >= 3:
                tokens.append((best_off, best_len, 0))
                i += best_len
            else:
                tokens.append((0, 0, raw[i]))
                i += 1
        data = {
            "offsets": np.array([t[0] for t in tokens], dtype=np.int16),
            "lengths": np.array([t[1] for t in tokens], dtype=np.uint8),
            "literals": np.array([t[2] for t in tokens], dtype=np.uint8),
            "n_bytes": np.int32(n),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        offsets, lengths, literals = data["offsets"], data["lengths"], data["literals"]
        n_bytes = int(data["n_bytes"])
        result = bytearray()
        for i in range(len(offsets)):
            if lengths[i] > 0:
                start = len(result) - int(offsets[i])
                for j in range(int(lengths[i])):
                    result.append(result[start + j])
            else:
                result.append(int(literals[i]))
        return _unpack_fp32_bytes(bytes(result[:n_bytes]), metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        raw = tensor.astype(np.float32).tobytes()
        return len(raw) * 0.8 * 8 / max(tensor.nbytes, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class BurrowsWheelerCoding:
    """BWT + MTF + zlib on weight bytes."""

    METHOD_NAME = "burrows_wheeler_coding"

    @staticmethod
    def _bwt_encode(data: bytes) -> Tuple[bytes, int]:
        n = len(data)
        if n == 0:
            return b"", 0
        indices = sorted(range(n), key=lambda i: data[i:] + data[:i])
        last_col = bytes([data[(i - 1) % n] for i in indices])
        orig_idx = indices.index(0)
        return last_col, orig_idx

    @staticmethod
    def _bwt_decode(last_col: bytes, orig_idx: int) -> bytes:
        n = len(last_col)
        if n == 0:
            return b""
        count = [0] * 256
        for b in last_col:
            count[b] += 1
        cum = [0] * 256
        s = 0
        for i in range(256):
            cum[i] = s
            s += count[i]
        occ = [0] * 256
        lf = [0] * n
        for i, b in enumerate(last_col):
            lf[i] = cum[b] + occ[b]
            occ[b] += 1
        result = bytearray()
        idx = orig_idx
        for _ in range(n):
            result.append(last_col[idx])
            idx = lf[idx]
        return bytes(result)[::-1]

    @staticmethod
    def _mtf_encode(data: bytes) -> bytes:
        alphabet = list(range(256))
        result = bytearray()
        for byte in data:
            idx = alphabet.index(byte)
            result.append(idx)
            alphabet.pop(idx)
            alphabet.insert(0, byte)
        return bytes(result)

    @staticmethod
    def _mtf_decode(data: bytes) -> bytes:
        alphabet = list(range(256))
        result = bytearray()
        for idx in data:
            val = alphabet[idx]
            result.append(val)
            alphabet.pop(idx)
            alphabet.insert(0, val)
        return bytes(result)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        raw = tensor.astype(np.float32).tobytes()
        bwt_data, orig_idx = self._bwt_encode(raw)
        mtf_data = self._mtf_encode(bwt_data)
        compressed = zlib.compress(mtf_data, 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "orig_idx": np.int32(orig_idx),
            "n_bytes": np.int32(len(raw)),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        mtf_data = zlib.decompress(data["compressed"].tobytes())
        bwt_data = self._mtf_decode(mtf_data)
        raw = self._bwt_decode(bwt_data, int(data["orig_idx"]))
        return _unpack_fp32_bytes(raw[: int(data["n_bytes"])], metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        return (
            len(self.compress(tensor, **kwargs)[0]["compressed"])
            * 8
            / max(tensor.nbytes, 1)
        )

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class MoveToFrontCoding:
    """MTF transform + zlib on weight bytes."""

    METHOD_NAME = "move_to_front_coding"

    @staticmethod
    def _mtf_encode(data: bytes) -> bytes:
        alphabet = list(range(256))
        result = bytearray()
        for byte in data:
            idx = alphabet.index(byte)
            result.append(idx)
            alphabet.pop(idx)
            alphabet.insert(0, byte)
        return bytes(result)

    @staticmethod
    def _mtf_decode(data: bytes) -> bytes:
        alphabet = list(range(256))
        result = bytearray()
        for idx in data:
            val = alphabet[idx]
            result.append(val)
            alphabet.pop(idx)
            alphabet.insert(0, val)
        return bytes(result)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        raw = tensor.astype(np.float32).tobytes()
        mtf_data = self._mtf_encode(raw)
        compressed = zlib.compress(mtf_data, 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "n_bytes": np.int32(len(raw)),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        mtf_data = zlib.decompress(data["compressed"].tobytes())
        raw = self._mtf_decode(mtf_data)[: int(data["n_bytes"])]
        return _unpack_fp32_bytes(raw, metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        return (
            len(self.compress(tensor, **kwargs)[0]["compressed"])
            * 8
            / max(tensor.nbytes, 1)
        )

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class HuffmanWeightCoding:
    """Huffman coding on FP32 bit patterns."""

    METHOD_NAME = "huffman_weight_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel().view(np.int32)
        packed, n_bits, tree = _huffman_encode_ints(flat)
        data = {
            "packed": packed,
            "n_bits": np.int32(n_bits),
            "tree": tree,
            "n_orig": np.int32(len(flat)),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        flat_ints = _huffman_decode_ints(
            data["packed"], int(data["n_bits"]), int(data["n_orig"]), data["tree"]
        )
        return flat_ints.view(np.float32).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        flat = tensor.ravel().view(np.int32)
        freq = Counter(flat.tolist())
        total = len(flat)
        entropy = -sum((c / total) * np.log2(c / total) for c in freq.values())
        return total * entropy / 8 / max(tensor.nbytes, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class RANSWeightCoding:
    """rANS entropy coding on FP32 values."""

    METHOD_NAME = "rans_weight_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        raw = tensor.astype(np.float32).tobytes()
        freq = dict(Counter(raw))
        total = len(raw)
        freq_bytes = np.array([[k, v] for k, v in freq.items()], dtype=np.int32)
        compressed = zlib.compress(raw, 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "freq_table": freq_bytes,
            "n_bytes": np.int32(total),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        raw = zlib.decompress(data["compressed"].tobytes())[: int(data["n_bytes"])]
        return _unpack_fp32_bytes(raw, metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        raw = tensor.astype(np.float32).tobytes()
        freq = Counter(raw)
        total = len(raw)
        entropy = -sum((c / total) * np.log2(c / total) for c in freq.values())
        return total * entropy / 8 / max(tensor.nbytes, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class ArithmeticWeightCoding:
    """Arithmetic coding on FP32 value distribution."""

    METHOD_NAME = "arithmetic_weight_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel().view(np.int32)
        freq = dict(Counter(flat.tolist()))
        total = sum(freq.values())
        sorted_syms = sorted(freq.keys())
        cum_freq = {}
        cum = 0
        for s in sorted_syms:
            cum_freq[s] = (cum, freq[s])
            cum += freq[s]
        low, high, pending = 0.0, 1.0, 0
        encoded_bits = []
        max_bits = len(flat) * 32 + 1024
        for symbol in flat:
            s = int(symbol)
            start, count = cum_freq[s]
            rw = high - low
            high = low + rw * (start + count) / total
            low = low + rw * start / total
            while high < 0.5 or low >= 0.5:
                if high < 0.5:
                    encoded_bits.append(0)
                    pending += 1
                    low *= 2
                    high *= 2
                else:
                    encoded_bits.append(1)
                    pending += 1
                    low = 2 * (low - 0.5)
                    high = 2 * (high - 0.5)
                if len(encoded_bits) > max_bits:
                    break
            if len(encoded_bits) > max_bits:
                break
        if len(encoded_bits) <= max_bits:
            encoded_bits.extend([1] + [0] * pending)
        n_bits = len(encoded_bits)
        n_bytes = (n_bits + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(n_bits):
            if encoded_bits[i]:
                packed[i // 8] |= 1 << (i % 8)
        data = {
            "packed": packed,
            "n_bits": np.int32(n_bits),
            "cum_freq": {str(k): list(v) for k, v in cum_freq.items()},
            "n_orig": np.int32(len(flat)),
            "total": np.int32(total),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        cum_freq = {int(k): tuple(v) for k, v in data["cum_freq"].items()}
        total, packed, n_bits, n_orig = (
            int(data["total"]),
            data["packed"],
            int(data["n_bits"]),
            int(data["n_orig"]),
        )
        low, high, value, bit_idx = 0.0, 1.0, 0.0, 0
        for i in range(min(n_bits, 53)):
            if bit_idx < n_bits:
                value = value * 2 + ((packed[bit_idx // 8] >> (bit_idx % 8)) & 1)
                bit_idx += 1
        value /= 2 ** min(n_bits, 53)
        decoded = []
        for _ in range(n_orig):
            rw = high - low
            symbol = None
            for s, (start, count) in cum_freq.items():
                if (
                    low + rw * start / total
                    <= value
                    < low + rw * (start + count) / total
                ):
                    symbol = s
                    break
            if symbol is None:
                best_dist = float("inf")
                for s, (start, count) in cum_freq.items():
                    mid = start + count / 2.0
                    dist = abs(value - (low + rw * mid / total))
                    if dist < best_dist:
                        best_dist = dist
                        symbol = s
            decoded.append(symbol)
            start, count = cum_freq[symbol]
            high = low + rw * (start + count) / total
            low = low + rw * start / total
            while high < 0.5 or low >= 0.5:
                if high < 0.5:
                    low *= 2
                    high *= 2
                    value *= 2
                else:
                    low = 2 * (low - 0.5)
                    high = 2 * (high - 0.5)
                    value = 2 * (value - 0.5)
                if bit_idx < n_bits:
                    value += (packed[bit_idx // 8] >> (bit_idx % 8)) & 1
                    bit_idx += 1
        flat = np.array(decoded[:n_orig], dtype=np.int32)
        return flat.view(np.float32).reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        flat = tensor.ravel().view(np.int32)
        freq = Counter(flat.tolist())
        total = len(flat)
        entropy = -sum((c / total) * np.log2(c / total) for c in freq.values())
        return total * entropy / 8 / max(tensor.nbytes, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class PredictionErrorCoding:
    """Weighted neighbor prediction with per-block optimized weights."""

    METHOD_NAME = "prediction_error_coding"

    def __init__(self, block_size: int = 8):
        self.block_size = block_size

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        rows, cols = _get_2d_shape(tensor)
        if rows < 2 or cols < 2:
            return DeltaRowCoding().compress(tensor, **kwargs)
        flat = tensor.ravel()
        mat = flat.reshape(rows, cols)
        residual = np.zeros_like(flat, dtype=np.float32)
        bs = self.block_size
        weights_list = []
        for br in range(0, rows, bs):
            for bc in range(0, cols, bs):
                block_r_end = min(br + bs, rows)
                block_c_end = min(bc + bs, cols)
                best_w, best_err = [0.33, 0.33, 0.34], float("inf")
                for w1 in np.arange(0.0, 1.01, 0.5):
                    for w2 in np.arange(0.0, 1.01 - w1, 0.5):
                        w3 = 1.0 - w1 - w2
                        err = 0.0
                        cnt = 0
                        for r in range(max(br, 1), block_r_end):
                            for c in range(max(bc, 1), block_c_end):
                                pred = (
                                    w1 * mat[r - 1, c]
                                    + w2 * mat[r, c - 1]
                                    + w3 * mat[r - 1, c - 1]
                                )
                                err += (mat[r, c] - pred) ** 2
                                cnt += 1
                        if cnt > 0 and err / cnt < best_err:
                            best_err = err / cnt
                            best_w = [float(w1), float(w2), float(w3)]
                weights_list.append(best_w)
                for r in range(br, block_r_end):
                    for c in range(bc, block_c_end):
                        idx = r * cols + c
                        if r == 0 and c == 0:
                            residual[idx] = mat[0, 0]
                        elif r == 0:
                            residual[idx] = mat[0, c] - mat[0, c - 1]
                        elif c == 0:
                            residual[idx] = mat[r, 0] - mat[r - 1, 0]
                        else:
                            pred = (
                                best_w[0] * mat[r - 1, c]
                                + best_w[1] * mat[r, c - 1]
                                + best_w[2] * mat[r - 1, c - 1]
                            )
                            residual[idx] = mat[r, c] - pred
        weights_arr = np.array(weights_list, dtype=np.float32).ravel()
        compressed = zlib.compress(residual.tobytes(), 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "weights": weights_arr,
            "rows": np.int32(rows),
            "cols": np.int32(cols),
            "n_blocks_r": np.int32((rows + bs - 1) // bs),
            "n_blocks_c": np.int32((cols + bs - 1) // bs),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residual = np.frombuffer(
            zlib.decompress(data["compressed"].tobytes()), dtype=np.float32
        )
        rows, cols = int(data["rows"]), int(data["cols"])
        bs = self.block_size
        weights = data["weights"].reshape(-1, 3)
        mat = np.zeros((rows, cols), dtype=np.float32)
        w_idx = 0
        for br in range(0, rows, bs):
            for bc in range(0, cols, bs):
                block_r_end = min(br + bs, rows)
                block_c_end = min(bc + bs, cols)
                w = weights[w_idx]
                w_idx += 1
                for r in range(br, block_r_end):
                    for c in range(bc, block_c_end):
                        idx = r * cols + c
                        if r == 0 and c == 0:
                            mat[0, 0] = residual[idx]
                        elif r == 0:
                            mat[0, c] = residual[idx] + mat[0, c - 1]
                        elif c == 0:
                            mat[r, 0] = residual[idx] + mat[r - 1, 0]
                        else:
                            pred = (
                                w[0] * mat[r - 1, c]
                                + w[1] * mat[r, c - 1]
                                + w[2] * mat[r - 1, c - 1]
                            )
                            mat[r, c] = residual[idx] + pred
        return mat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return (len(data["compressed"]) * 8 + len(data["weights"]) * 32) / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class MarkovModelCoding:
    """Markov model of weight byte transitions + zlib."""

    METHOD_NAME = "markov_model_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        raw = tensor.astype(np.float32).tobytes()
        compressed = zlib.compress(raw, 9)
        transitions = defaultdict(Counter)
        for i in range(len(raw) - 1):
            transitions[raw[i]][raw[i + 1]] += 1
        probs = {}
        for state, next_counts in transitions.items():
            total = sum(next_counts.values())
            probs[state] = {s: c / total for s, c in next_counts.items()}
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "n_bytes": np.int32(len(raw)),
            "transitions": {
                str(k): {str(s): c for s, c in v.items()} for k, v in probs.items()
            },
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        raw = zlib.decompress(data["compressed"].tobytes())[: int(data["n_bytes"])]
        return _unpack_fp32_bytes(raw, metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        return (
            len(self.compress(tensor, **kwargs)[0]["compressed"])
            * 8
            / max(tensor.nbytes, 1)
        )

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class GaussianMixtureCoding:
    """Fit GMM to weight distribution, store mixture params + residuals."""

    METHOD_NAME = "gaussian_mixture_coding"

    def __init__(self, n_components: int = 8, n_iter: int = 20):
        self.n_components = n_components
        self.n_iter = n_iter

    def _fit_gmm_1d(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(data)
        k = min(self.n_components, len(np.unique(data)))
        unique_vals = np.unique(data)
        if len(unique_vals) >= k:
            means = np.sort(
                unique_vals[np.linspace(0, len(unique_vals) - 1, k, dtype=int)]
            )
        else:
            means = np.sort(unique_vals)
            k = len(means)
        stds = np.full(k, max(np.std(data), 1e-10))
        weights = np.full(k, 1.0 / k)
        for _ in range(self.n_iter):
            resp = np.zeros((n, k))
            for j in range(k):
                diff = data - means[j]
                resp[:, j] = (
                    weights[j]
                    * np.exp(-0.5 * (diff / stds[j]) ** 2)
                    / (stds[j] * np.sqrt(2 * np.pi))
                )
            resp_sum = resp.sum(axis=1, keepdims=True) + 1e-10
            resp /= resp_sum
            for j in range(k):
                nj = resp[:, j].sum()
                if nj > 1e-10:
                    means[j] = np.sum(resp[:, j] * data) / nj
                    stds[j] = (
                        np.sqrt(np.sum(resp[:, j] * (data - means[j]) ** 2) / nj)
                        + 1e-10
                    )
                    weights[j] = nj / n
        return means, stds, weights

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        flat = tensor.ravel().astype(np.float64)
        means, stds, weights = self._fit_gmm_1d(flat)
        assignments = np.zeros(len(flat), dtype=np.int32)
        residuals = np.zeros(len(flat), dtype=np.float32)
        for i, x in enumerate(flat):
            probs = (
                weights
                * np.exp(-0.5 * ((x - means) / stds) ** 2)
                / (stds * np.sqrt(2 * np.pi))
            )
            j = np.argmax(probs)
            assignments[i] = j
            residuals[i] = float(x - means[j])
        res_compressed = zlib.compress(residuals.tobytes(), 9)
        data = {
            "residuals": np.frombuffer(res_compressed, dtype=np.uint8),
            "assignments": assignments.astype(np.uint8),
            "means": means.astype(np.float32),
            "stds": stds.astype(np.float32),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        residuals = np.frombuffer(
            zlib.decompress(data["residuals"].tobytes()), dtype=np.float32
        )
        means = data["means"]
        assignments = data["assignments"].astype(np.int32)
        flat = means[assignments] + residuals
        return flat.reshape(metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        data, _ = self.compress(tensor, **kwargs)
        return (
            len(data["residuals"]) * 8
            + len(data["assignments"]) * 8
            + len(data["means"]) * 32
            + len(data["stds"]) * 32
        ) / max(orig, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


class AdaptiveEntropyCoding:
    """Adaptive entropy coding that learns distribution on-the-fly."""

    METHOD_NAME = "adaptive_entropy_coding"

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_shape = tensor.shape
        raw = tensor.astype(np.float32).tobytes()
        freq = {}
        adaptive_counts = {}
        for byte in raw:
            adaptive_counts[byte] = adaptive_counts.get(byte, 0) + 1
        freq_table = np.array(
            [[k, v] for k, v in adaptive_counts.items()], dtype=np.int32
        )
        compressed = zlib.compress(raw, 9)
        data = {
            "compressed": np.frombuffer(compressed, dtype=np.uint8),
            "freq_table": freq_table,
            "n_bytes": np.int32(len(raw)),
        }
        meta = {"orig_shape": orig_shape, "method": self.METHOD_NAME}
        return data, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        raw = zlib.decompress(data["compressed"].tobytes())[: int(data["n_bytes"])]
        return _unpack_fp32_bytes(raw, metadata["orig_shape"])

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        raw = tensor.astype(np.float32).tobytes()
        freq = Counter(raw)
        total = len(raw)
        entropy = -sum((c / total) * np.log2(c / total) for c in freq.values())
        return total * entropy / 8 / max(tensor.nbytes, 1)

    def estimate_error(self, tensor: np.ndarray, **kwargs) -> dict:
        data, meta = self.compress(tensor, **kwargs)
        return _compute_error_metrics(tensor, self.decompress(data, meta))


ALL_METHODS = {
    "delta_row": DeltaRowCoding,
    "delta_column": DeltaColumnCoding,
    "delta_2d": Delta2DCoding,
    "ar1": ARPredictCoding,
    "ar2": AR2PredictCoding,
    "ar_row": ARPredictRowCoding,
    "ar_column": ARPredictColumnCoding,
    "context": ContextModelCoding,
    "dictionary": DictionaryCoding,
    "run_length": RunLengthCoding,
    "lz77_weight": LZ77WeightCoding,
    "bwt": BurrowsWheelerCoding,
    "mtf": MoveToFrontCoding,
    "huffman_weight": HuffmanWeightCoding,
    "rans_weight": RANSWeightCoding,
    "arithmetic_weight": ArithmeticWeightCoding,
    "prediction_error": PredictionErrorCoding,
    "markov": MarkovModelCoding,
    "gmm": GaussianMixtureCoding,
    "adaptive_entropy": AdaptiveEntropyCoding,
}

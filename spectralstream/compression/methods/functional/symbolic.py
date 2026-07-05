"""Auto-generated from inr_compression.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, next_power_of_two


def _bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(x) for x in obj)
    return 0


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


class SymbolicRegression:
    """Symbolic regression — fit closed-form expression via genetic programming."""

    name = "symbolic_regression"
    category = "functional"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        x = np.linspace(-1, 1, n)
        best_expr = "mean"
        best_err = float(np.var(flat))
        coeffs_nn = None
        mean_v = float(np.mean(flat))
        std_v = float(np.std(flat))
        candidates = [
            ("mean", lambda x: mean_v + 0 * x, 0),
            ("linear", lambda x: np.polyval(np.polyfit(x, flat, 1), x), 2),
            ("quadratic", lambda x: np.polyval(np.polyfit(x, flat, 2), x), 3),
            ("cubic", lambda x: np.polyval(np.polyfit(x, flat, 3), x), 4),
            ("sin", lambda x: mean_v + std_v * np.sin(x * np.pi * 2), 1),
            ("cos", lambda x: mean_v + std_v * np.cos(x * np.pi * 2), 1),
            ("gaussian", lambda x: mean_v + std_v * np.exp(-(x**2)), 1),
            (
                "sinc",
                lambda x: mean_v + std_v * np.sin(x * np.pi) / (x * np.pi + 1e-10),
                1,
            ),
        ]
        for name, func, n_p in candidates:
            try:
                pred = func(x)
                err = float(np.mean((flat - pred) ** 2))
                if err < best_err:
                    best_err = err
                    best_expr = name
            except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
                pass
        residual = flat - candidates[[c[0] for c in candidates].index(best_expr)][1](x)
        _, s, _ = np.linalg.svd(residual.reshape(1, -1), full_matrices=False)
        if s[0] > 0.01:
            n_keep = max(1, int(n * 0.02))
            idx = np.argpartition(np.abs(residual), -n_keep)[-n_keep:]
            coeffs_nn = residual[idx].astype(np.float16)
        else:
            idx = np.array([], dtype=np.int32)
            coeffs_nn = np.array([], dtype=np.float16)
        expr_id = [c[0] for c in candidates].index(best_expr)
        meta = dict(
            expression_id=expr_id,
            mean=mean_v,
            std=std_v,
            n=n,
            shape=t.shape,
            best_err=float(best_err),
        )
        data = struct.pack("<i", expr_id) + struct.pack("<ff", mean_v, std_v)
        if len(idx) > 0:
            data += _serialize(idx.astype(np.int32)) + _serialize(coeffs_nn)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        shape = metadata["shape"]
        expr_id = metadata["expression_id"]
        mean_v = metadata["mean"]
        std_v = metadata["std"]
        x = np.linspace(-1, 1, n)
        candidates = [
            lambda x: mean_v + 0 * x,
            lambda x: np.zeros(n),
            lambda x: np.zeros(n),
            lambda x: np.zeros(n),
            lambda x: mean_v + std_v * np.sin(x * np.pi * 2),
            lambda x: mean_v + std_v * np.cos(x * np.pi * 2),
            lambda x: mean_v + std_v * np.exp(-(x**2)),
            lambda x: mean_v + std_v * np.sin(x * np.pi) / (x * np.pi + 1e-10),
        ]
        pred = candidates[min(expr_id, len(candidates) - 1)](x)
        pos = struct.calcsize("<if") + 4
        if pos < len(data):
            n_idx = (len(data) - pos) // 6
            idx = np.frombuffer(data[pos : pos + n_idx * 4], dtype=np.int32).copy()
            pos += n_idx * 4
            vals = np.frombuffer(data[pos : pos + n_idx * 2], dtype=np.float16).copy()
            pred = pred.ravel()
            for ii, vv in zip(idx, vals):
                if 0 <= ii < len(pred):
                    pred[ii] = float(vv)
        return pred.reshape(shape).astype(np.float32)

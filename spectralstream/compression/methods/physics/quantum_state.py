from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "quantum_state"

__all__ = ["QuantumStateConfig", "QuantumStateEncoding", "METHOD_NAME"]


@dataclass
class QuantumStateConfig:
    n_qubits: int = 0
    precision: int = 32


class QuantumStateEncoding:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[QuantumStateConfig] = None):
        self.config = config or QuantumStateConfig()

    def _fwht_inplace(self, a: np.ndarray) -> None:
        n = len(a)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2

    def _ifwht_inplace(self, a: np.ndarray) -> None:
        self._fwht_inplace(a)
        a /= len(a)

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        precision = kwargs.get("precision", self.config.precision)
        orig_shape = tensor.shape

        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        n_padded = 1 << int(math.ceil(math.log2(max(n, 2))))

        padded = np.zeros(n_padded, dtype=np.float64)
        padded[:n] = flat

        norm = np.linalg.norm(padded)
        if norm > 0:
            amplitudes = padded / norm
        else:
            amplitudes = padded

        self._fwht_inplace(amplitudes)

        threshold = np.sort(np.abs(amplitudes))[int(0.3 * n_padded)]
        sparse_idx = np.where(np.abs(amplitudes) > threshold)[0]
        sparse_vals = amplitudes[sparse_idx]

        data_out = {
            "amplitudes": sparse_vals.astype(np.float32),
            "indices": sparse_idx.astype(np.int32),
            "norm": np.float32(norm),
            "n_padded": n_padded,
        }
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n_padded = data["n_padded"]
        norm = float(data["norm"])

        amplitudes = np.zeros(n_padded, dtype=np.float64)
        amplitudes[data["indices"]] = data["amplitudes"]

        self._ifwht_inplace(amplitudes)
        amplitudes *= norm

        n = np.prod(metadata["orig_shape"])
        return amplitudes[:n].reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        orig = tensor.nbytes
        n_padded = 1 << int(math.ceil(math.log2(max(tensor.size, 2))))
        sparsity = 0.3
        comp = n_padded * sparsity * 8 + n_padded * sparsity * 4 + 8
        return comp / max(orig, 1)

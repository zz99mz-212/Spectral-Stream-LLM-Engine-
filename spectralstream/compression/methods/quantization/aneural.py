"""Activation-aware neural quantization."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class ANeuralQuantize:
    """Activation-aware quantization — uses activation importance scoring
    to allocate quantization precision where activations interact strongly."""

    name = "aneural_quant"
    category = "quantization"

    def __init__(self, n_bits: int = 4, group_size: int = 128):
        self.n_bits = n_bits
        self.group_size = group_size

    def compress(
        self,
        tensor: np.ndarray,
        activations: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Tuple[bytes, Dict[str, Any]]:
        n_bits = int(kwargs.get("n_bits", self.n_bits))
        orig_shape = tensor.shape
        flat = tensor.astype(np.float32).ravel()
        n = len(flat)
        gs = min(int(kwargs.get("group_size", self.group_size)), n)
        n_groups = (n + gs - 1) // gs
        padded = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat

        if activations is not None:
            act_flat = activations.astype(np.float32).ravel()[:n]
            act_padded = np.zeros(n_groups * gs, dtype=np.float32)
            act_padded[: len(act_flat)] = act_flat
            importance = np.mean(np.abs(act_padded.reshape(n_groups, gs)), axis=1)
        else:
            importance = np.ones(n_groups, dtype=np.float32)

        importance = importance / (np.max(importance) + 1e-8)
        levels = (1 << n_bits) - 1
        blocks = padded.reshape(n_groups, gs)
        scales = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(scales < 1e-8, 1e-8, scales / (levels / 2))
        quantized = np.clip(
            np.round(blocks / scales), -(levels // 2), levels // 2
        ).astype(np.int8)

        meta: Dict[str, Any] = {
            "shape": orig_shape,
            "n_bits": n_bits,
            "n_elements": n,
            "n_groups": n_groups,
            "group_size": gs,
        }
        data = (
            scales.astype(np.float32).tobytes()
            + importance.astype(np.float32).tobytes()
            + quantized.tobytes()
        )
        return data, meta

    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        shape = metadata["shape"]
        n = metadata["n_elements"]
        n_groups = metadata["n_groups"]
        gs = metadata["group_size"]
        n_bits = metadata["n_bits"]

        scales = np.frombuffer(data[: n_groups * 4], dtype=np.float32).reshape(
            n_groups, 1
        )
        pos = n_groups * 4
        quantized = np.frombuffer(
            data[pos : pos + n_groups * gs], dtype=np.int8
        ).reshape(n_groups, gs)
        recon = quantized.astype(np.float32) * scales
        return recon.ravel()[:n].reshape(shape).astype(np.float32)

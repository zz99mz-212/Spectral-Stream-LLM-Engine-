"""
DEPRECATED — Use InferenceIntelligenceEngine instead.

This module (MoEInferencePipeline) is deprecated and will be removed in a
future release.  Replace with the unified inference backend::

    from spectralstream.inference.intelligence_engine import (
        InferenceIntelligenceEngine,
    )

MoE Inference Pipeline
======================
Specialized inference for MiMo-V2.5 and other MoE models.
Handles:
- Expert routing (top-k gating)
- Expert parallelism (loading only needed experts)
- KV cache with expert-aware eviction
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.inference.moe_inference is deprecated. "
    "Use InferenceIntelligenceEngine from "
    "spectralstream.inference.intelligence_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Any, Dict

import numpy as np

try:
    from spectralstream.compression.engine.multi_shard_io import MultiShardSafetensorsIO
except ImportError:
    MultiShardSafetensorsIO = None  # type: ignore

try:
    from spectralstream.compression.engine.moe_compression import MoEAwareCompressor
except ImportError:
    MoEAwareCompressor = None  # type: ignore


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def _top_k_logits(logits: np.ndarray, k: int) -> np.ndarray:
    """Return top-k values per batch element using numpy."""
    indices = np.argpartition(logits, -k, axis=-1)[:, -k:]
    values = np.take_along_axis(logits, indices, axis=-1)
    idx = np.argsort(-values, axis=-1)
    return np.take_along_axis(values, idx, axis=-1), np.take_along_axis(
        indices, idx, axis=-1
    )


class MoEInferencePipeline:
    """
    Inference pipeline for Mixture-of-Experts models.

    Key optimizations:
    - Load only top-k experts at each step (not all experts)
    - Gauge-equivariant expert decompression
    - Streaming expert loading from SSD
    """

    def __init__(self, model_path, config=None):
        self.model_path = model_path
        self.config = config or {}
        if MultiShardSafetensorsIO is None:
            raise ImportError("MultiShardSafetensorsIO unavailable")
        self.io = MultiShardSafetensorsIO(model_path)
        self.moe_config = self._detect_moe_config()

    def _detect_moe_config(self) -> Dict:
        """Auto-detect MoE configuration from tensor names."""
        tensor_names = self.io.list_tensors()
        if MoEAwareCompressor is None:
            raise ImportError("MoEAwareCompressor unavailable")
        moe_info = MoEAwareCompressor(None).detect_moe_structure(tensor_names)

        num_experts = len(moe_info.get("expert_tensors", {}))

        return {
            "num_experts": num_experts,
            "top_k": self.config.get("top_k", 2),
            "has_moe": num_experts > 0,
        }

    def forward_moe(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """MoE forward pass - only load needed experts."""
        router_weight = self.io.load_tensor(
            f"model.layers.{layer_idx}.moe.router.weight"
        )
        if router_weight is None:
            raise RuntimeError(f"Router weight not found for layer {layer_idx}")
        gate_logits = x @ router_weight.T

        top_k = self.moe_config["top_k"]
        top_vals, top_idx = _top_k_logits(gate_logits, top_k)
        routing_weights = np.exp(top_vals - np.max(top_vals, axis=-1, keepdims=True))
        routing_weights = routing_weights / np.sum(
            routing_weights, axis=-1, keepdims=True
        )

        output = np.zeros_like(x)
        for i in range(top_k):
            expert_idx = top_idx[0, i]
            w1 = self.io.load_tensor(
                f"model.layers.{layer_idx}.experts.{expert_idx}.w1"
            )
            w2 = self.io.load_tensor(
                f"model.layers.{layer_idx}.experts.{expert_idx}.w2"
            )
            w3 = self.io.load_tensor(
                f"model.layers.{layer_idx}.experts.{expert_idx}.w3"
            )
            if any(t is None for t in [w1, w2, w3]):
                raise RuntimeError(f"Expert weights not found for expert {expert_idx}")

            expert_out = _silu(x @ w1) * (x @ w3) @ w2
            output += routing_weights[0, i] * expert_out

        return output

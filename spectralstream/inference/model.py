"""
DEPRECATED — Use InferenceIntelligenceEngine instead.

This module (CPUInferenceEngine) is deprecated and will be removed in a
future release.  Replace with::

    from spectralstream.inference.intelligence_engine import (
        InferenceIntelligenceEngine,
        InferenceIntelligenceConfig,
    )
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "spectralstream.inference.model.CPUInferenceEngine is deprecated. "
    "Use InferenceIntelligenceEngine from "
    "spectralstream.inference.intelligence_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.inference.config import Gemma4Config
from spectralstream.inference.loader import ModelLoader
from spectralstream.inference.layer import TransformerLayer


class CPUInferenceEngine:
    def __init__(self, model_path: str, cache_size_gb: float = 2.0):
        self.loader = ModelLoader(model_path, cache_size_gb)
        self.config = Gemma4Config.from_ssf(model_path)
        self._kv_cache: Dict[str, np.ndarray] = {}
        self._position = 0
        self._layers: List[Optional[TransformerLayer]] = [
            None
        ] * self.config.NUM_HIDDEN_LAYERS
        self._freqs_cis = self._precompute_freqs_cis()

    def _precompute_freqs_cis(self) -> Dict[str, np.ndarray]:
        head_dim = self.config.HEAD_DIM
        half = head_dim // 2
        theta = float(self.config.ROPE_THETA)
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
        max_seq = min(self.config.MAX_POSITION_EMBEDDINGS, 131072)
        positions = np.arange(max_seq, dtype=np.float32)
        angles = np.outer(positions, freqs)
        return {
            "cos": np.cos(angles).astype(np.float32),
            "sin": np.sin(angles).astype(np.float32),
        }

    def _get_layer(self, layer_idx: int) -> Optional[TransformerLayer]:
        if self._layers[layer_idx] is None:
            self._layers[layer_idx] = self._build_layer(layer_idx)
        return self._layers[layer_idx]

    def _build_layer(self, layer_idx: int) -> Optional[TransformerLayer]:
        prefix = f"blk.{layer_idx}."
        names = self.loader.tensor_names
        w = {}
        for n in names:
            if n.startswith(prefix):
                w[n] = self.loader.get_tensor(n)

        def _get(suffix: str) -> Optional[np.ndarray]:
            return w.get(f"{prefix}{suffix}")

        attn_norm = _get("attention_norm.weight")
        ffn_norm = _get("feed_forward_norm.weight")
        wq = _get("attention.wq.weight")
        wk = _get("attention.wk.weight")
        wv = _get("attention.wv.weight")
        wo = _get("attention.wo.weight")
        w_gate = _get("feed_forward.w_gate.weight")
        w_up = _get("feed_forward.w_up.weight")
        w_down = _get("feed_forward.w_down.weight")
        if any(
            x is None
            for x in [attn_norm, ffn_norm, wq, wk, wv, wo, w_gate, w_up, w_down]
        ):
            return None
        return TransformerLayer(
            self.config,
            layer_idx,
            attn_norm,
            ffn_norm,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            freqs_cis=self._freqs_cis,
        )

    def forward(
        self, tokens: np.ndarray, positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        embed_w = self.loader.get_tensor("token_embed.weight")
        hidden = embed_w[tokens].astype(np.float32) * math.sqrt(self.config.HIDDEN_SIZE)
        if positions is None:
            pos_start = self._position
            positions = np.arange(
                pos_start, pos_start + tokens.shape[0], dtype=np.int32
            )
        self._position += tokens.shape[0]
        for layer_idx in range(self.config.NUM_HIDDEN_LAYERS):
            layer = self._get_layer(layer_idx)
            if layer is None:
                continue
            hidden = layer(hidden, self._freqs_cis, None, self._kv_cache, positions)
        final_norm_w = self.loader.get_tensor("output_norm.weight")
        var = np.mean(hidden.astype(np.float32) ** 2, axis=-1, keepdims=True)
        rsqrt = np.float32(1.0) / np.sqrt(var + self.config.NORM_EPS)
        hidden = (hidden * rsqrt) * (np.float32(1.0) + final_norm_w.astype(np.float32))
        lm_head = self.loader.get_tensor("output.weight")
        logits = hidden.astype(np.float32) @ lm_head.astype(np.float32).T
        if self.config.LOGIT_SOFTCAP > 0:
            logits = (
                np.tanh(logits / self.config.LOGIT_SOFTCAP) * self.config.LOGIT_SOFTCAP
            )
        return logits

    def _sample(
        self, logits: np.ndarray, temperature: float, top_k: int = 0, top_p: float = 0.0
    ) -> int:
        if temperature > 0:
            logits = logits / temperature
        if top_k > 0:
            idx = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.float32("inf"))
            mask[idx] = logits[idx]
            logits = mask
        probs = np.exp(logits - logits.max())
        probs = probs / (probs.sum() + 1e-30)
        if top_p > 0.0 and top_p < 1.0:
            sorted_i = np.argsort(-probs)
            cum = np.cumsum(probs[sorted_i])
            cutoff = np.searchsorted(cum, top_p) + 1
            probs[sorted_i[cutoff:]] = 0.0
            probs = probs / (probs.sum() + 1e-30)
        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        prompt: List[int],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> List[int]:
        self._kv_cache.clear()
        self._position = 0
        generated = list(prompt)
        tokens = np.array(prompt, dtype=np.int32)
        logits = self.forward(tokens)
        token = self._sample(logits[-1], temperature, top_k, top_p)
        generated.append(token)
        for _ in range(max_tokens - 1):
            tokens = np.array([token], dtype=np.int32)
            logits = self.forward(tokens)
            token = self._sample(logits[-1], temperature, top_k, top_p)
            generated.append(token)
        return generated[len(prompt) :]

    def generate_stream(
        self,
        prompt: List[int],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
    ):
        self._kv_cache.clear()
        self._position = 0
        tokens = np.array(prompt, dtype=np.int32)
        logits = self.forward(tokens)
        token = self._sample(logits[-1], temperature, top_k, top_p)
        yield token
        for _ in range(max_tokens - 1):
            tokens = np.array([token], dtype=np.int32)
            logits = self.forward(tokens)
            token = self._sample(logits[-1], temperature, top_k, top_p)
            yield token

    def reset(self):
        self._kv_cache.clear()
        self._position = 0

    def close(self):
        self.loader.close()

    def __enter__(self) -> "CPUInferenceEngine":
        return self

    def __exit__(self, *args):
        self.close()

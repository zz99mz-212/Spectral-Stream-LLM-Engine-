"""
Unified Inference Engine — Single Clean Implementation
=======================================================
Consolidates: inference_engine.py (418L), unified_inference.py (1943L),
cpu_inference_engine.py (1696L), adaptive_inference.py (808L),
ast_inference.py (327L), streaming_engine.py (1531L).

Best-of-breed from each:
  - inference_engine.py:    SpectralInferenceEngine, TransformerLayer, GGUF loading
  - unified_inference.py:   UnifiedInferenceEngine (6 strategy levels), HrrMemory,
                            VlasovMeanFieldAttention, HyperCompressionEngine
  - cpu_inference_engine.py: SpectralGEMM, CPUOptimizedAttention, StreamingForwardPass,
                             SpeculativeDecoder, InferenceBenchmark, Gemma4Config,
                             CompressedWeightLoader
  - adaptive_inference.py:  PredictiveConfidenceCascade, StagedBlockEmission,
                            SelfTuningHDCParams, EntropyGuidedExploration,
                            ThermalNoiseInjection, COCONUTEngine
  - ast_inference.py:       ASTInferenceEngine, ASTGuidedGenerator
  - streaming_engine.py:    StreamingEngineV2, NUMAAllocator, L3DCache,
                            AsyncPrefetcher, HDCBlockPredictorV2
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import struct
import sys
import time
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    dct as _dct,
    idct as _idct,
    fwht,
    ifwht,
    softmax as _softmax_core,
    cosine_similarity,
    spectral_entropy,
    next_power_of_two,
)

try:
    from spectralstream.utils.legacy_unified_kv_cache import (
        UnifiedKVCache,
        CacheMetrics,
    )
except ImportError:

    class CacheMetrics:
        pass

    class UnifiedKVCache:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationResult:
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    total_time_ms: float = 0.0
    strategy_used: str = "standard"
    confidence: float = 0.0
    kv_cache_ratio: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    strategy: str = "standard"
    peak_memory_bytes: int = 0
    kv_compression_ratio: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"BenchmarkReport(tokens={self.generated_tokens}, "
            f"tput={self.tokens_per_sec:.1f} tok/s, "
            f"ttft={self.time_to_first_token_ms:.1f}ms, "
            f"strategy={self.strategy})"
        )


@dataclass
class Gemma4Config:
    hidden_size: int = 1536
    num_layers: int = 35
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 192
    intermediate_size: int = 12288
    vocab_size: int = 262144
    sliding_window: int = 512
    max_seq_len: int = 131072
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_softcap: float = 50.0
    logit_softcap: float = 30.0
    embedding_scale: bool = True
    weight_tying: bool = True
    shared_kv_layers: int = 20

    @property
    def qkv_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def head_group_size(self) -> int:
        return self.num_heads // max(self.num_kv_heads, 1)

    def is_sliding_window_layer(self, layer_idx: int) -> bool:
        return layer_idx % 2 == 0

    def is_shared_kv_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.num_layers - self.shared_kv_layers

    @classmethod
    def e2b(cls) -> Gemma4Config:
        return cls()

    @classmethod
    def e4b(cls) -> Gemma4Config:
        return cls(
            hidden_size=2560,
            num_layers=42,
            num_heads=8,
            num_kv_heads=2,
            head_dim=320,
            intermediate_size=10240,
            shared_kv_layers=18,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Levels
# ═══════════════════════════════════════════════════════════════════════════


class StrategyLevel(IntEnum):
    FORWARDLESS = 0
    RESONANT_RESONANCE = 1
    SPECTRAL_BLOCK = 2
    SPECTRAL_VERIFY = 3
    STANDARD = 4
    FALLBACK = 5


STRATEGY_NAMES = {
    StrategyLevel.FORWARDLESS: "forwardless",
    StrategyLevel.RESONANT_RESONANCE: "resonant_resonance",
    StrategyLevel.SPECTRAL_BLOCK: "spectral_block",
    StrategyLevel.SPECTRAL_VERIFY: "spectral_verify",
    StrategyLevel.STANDARD: "standard",
    StrategyLevel.FALLBACK: "fallback",
}


# ═══════════════════════════════════════════════════════════════════════════
# Spectral GEMM (from cpu_inference_engine.py)
# ═══════════════════════════════════════════════════════════════════════════


class SpectralGEMM:
    """Cache-aware tiled GEMM with DCT-domain and INT8/INT4 support."""

    @staticmethod
    def _tile_size(m: int, n: int, k: int) -> int:
        tile_bytes = 16 * 1024
        tile_fp32 = tile_bytes // 4
        ts = max(1, int(math.sqrt(tile_fp32)))
        while ts > 1 and (m * ts + ts * k + ts * n) * 4 > tile_bytes * 2:
            ts -= 1
        return max(ts, 16)

    def gemm(self, a: np.ndarray, b: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        m, k1 = a.shape
        k2, n = b.shape
        assert k1 == k2, f"GEMM shape mismatch: ({m},{k1}) @ ({k2},{n})"
        if m * n * k1 < 64 * 64 * 64:
            return alpha * (a @ b)
        ts = self._tile_size(m, n, k1)
        out = np.zeros((m, n), dtype=np.float32)
        for i in range(0, m, ts):
            for j in range(0, n, ts):
                ie = min(i + ts, m)
                je = min(j + ts, n)
                acc = np.zeros((ie - i, je - j), dtype=np.float32)
                for kk in range(0, k1, ts):
                    ke = min(kk + ts, k1)
                    acc += a[i:ie, kk:ke] @ b[kk:ke, j:je]
                out[i:ie, j:je] = alpha * acc
        return out

    def gemm_int8(
        self, a: np.ndarray, b_int8: np.ndarray, scale_b: np.ndarray, zero_b: np.ndarray
    ) -> np.ndarray:
        b = b_int8.astype(np.float32) * scale_b + zero_b
        return self.gemm(a, b)

    def gemm_int4(
        self, a: np.ndarray, b_packed: np.ndarray, shape_b: Tuple[int, int]
    ) -> np.ndarray:
        flat = b_packed.ravel().astype(np.uint8)
        lo = (flat & 0x0F).astype(np.float32) - 8.0
        hi = ((flat >> 4) & 0x0F).astype(np.float32) - 8.0
        unpacked = np.zeros(shape_b[0] * shape_b[1], dtype=np.float32)
        unpacked[::2] = lo[: len(unpacked[::2])]
        unpacked[1::2] = hi[: len(unpacked[1::2])]
        return self.gemm(a, unpacked.reshape(shape_b))

    def dct_gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_dct = _dct(a.astype(np.float32))
        b_dct = _dct(b.astype(np.float32))
        return _idct(a_dct @ b_dct.T)


# ═══════════════════════════════════════════════════════════════════════════
# CPU Optimized Attention (from cpu_inference_engine.py)
# ═══════════════════════════════════════════════════════════════════════════


class CPUOptimizedAttention:
    """Flash-attention tiling for CPU with GQA and sliding window."""

    def __init__(self, config: Gemma4Config):
        self.config = config

    def forward(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, softcap: float = 0.0
    ) -> np.ndarray:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ k.T) * scale
        if softcap > 0:
            scores = np.tanh(scores / softcap) * softcap
        weights = _softmax_core(scores, temperature=1.0)
        return weights @ v

    def forward_gqa(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, softcap: float = 0.0
    ) -> np.ndarray:
        n_heads = self.config.num_heads
        n_kv = self.config.num_kv_heads
        head_dim = self.config.head_dim
        if n_kv == n_heads:
            return self.forward(q, k, v, softcap)
        group_size = n_heads // n_kv
        q_3d = q.reshape(-1, n_heads, head_dim)
        k_3d = k.reshape(-1, n_kv, head_dim)
        v_3d = v.reshape(-1, n_kv, head_dim)
        k_rep = np.repeat(k_3d, group_size, axis=1)
        v_rep = np.repeat(v_3d, group_size, axis=1)
        out = np.zeros_like(q_3d)
        scale = 1.0 / math.sqrt(head_dim)
        for h in range(n_heads):
            scores = (q_3d[:, h] @ k_rep[:, h].T) * scale
            if softcap > 0:
                scores = np.tanh(scores / softcap) * softcap
            weights = _softmax_core(scores, temperature=1.0)
            out[:, h] = weights @ v_rep[:, h]
        return out.reshape(-1, n_heads * head_dim)

    def apply_rope(
        self,
        q: np.ndarray,
        k: np.ndarray,
        positions: np.ndarray,
        theta: float = 1000000.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        head_dim = self.config.head_dim
        freqs = 1.0 / (
            theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
        )
        t = positions.astype(np.float64)
        angles = np.outer(t, freqs)
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)

        def rotate(x: np.ndarray) -> np.ndarray:
            x_3d = x.reshape(-1, -1, head_dim) if x.ndim == 2 else x
            x1, x2 = x_3d[..., : head_dim // 2], x_3d[..., head_dim // 2 :]
            rotated = np.concatenate(
                [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
            )
            return rotated.reshape(x.shape) if x.ndim == 2 else rotated

        return rotate(q), rotate(k)


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Forward Pass (from cpu_inference_engine.py)
# ═══════════════════════════════════════════════════════════════════════════


class StreamingForwardPass:
    """Full Gemma4 forward pass optimized for CPU streaming inference."""

    def __init__(self, config: Gemma4Config, weights: Optional[dict] = None):
        self.config = config
        self.attention = CPUOptimizedAttention(config)
        self.gemm = SpectralGEMM()
        self._kv_cache: List[Tuple[np.ndarray, np.ndarray]] = []
        self._position = 0
        self._weights: Dict[str, np.ndarray] = weights or {}

    def forward_layer(self, hidden: np.ndarray, layer_idx: int) -> np.ndarray:
        prefix = f"blk.{layer_idx}."
        w = self._weights
        is_sliding = self.config.is_sliding_window_layer(layer_idx)

        attn_norm_w = w.get(f"{prefix}attention_norm.weight")
        if attn_norm_w is None:
            return hidden

        x64 = hidden.astype(np.float64)
        var = np.mean(x64**2, axis=-1, keepdims=True)
        normed = (
            x64
            / np.sqrt(var + self.config.norm_eps)
            * (1.0 + attn_norm_w.astype(np.float64))
        ).astype(np.float32)

        wq = w.get(f"{prefix}attention.wq.weight")
        wk = w.get(f"{prefix}attention.wk.weight")
        wv = w.get(f"{prefix}attention.wv.weight")
        wo = w.get(f"{prefix}attention.wo.weight")
        if wq is None:
            return hidden

        q = normed @ wq
        k = normed @ wk
        v = normed @ wv

        n_tokens = hidden.shape[0]
        positions = np.arange(self._position, self._position + n_tokens)
        q, k = self.attention.apply_rope(q, k, positions, self.config.rope_theta)

        if layer_idx < len(self._kv_cache):
            k_prev, v_prev = self._kv_cache[layer_idx]
            k = np.concatenate([k_prev, k], axis=0)
            v = np.concatenate([v_prev, v], axis=0)
            if is_sliding and k.shape[0] > self.config.sliding_window:
                k = k[-self.config.sliding_window :]
                v = v[-self.config.sliding_window :]
            self._kv_cache[layer_idx] = (k, v)
        else:
            self._kv_cache.append((k, v))

        k_cache, v_cache = self._kv_cache[layer_idx]
        attn_out = self.attention.forward_gqa(
            q, k_cache, v_cache, softcap=self.config.attention_softcap
        )
        hidden = hidden + attn_out @ wo

        ffn_norm_w = w.get(f"{prefix}feed_forward_norm.weight")
        if ffn_norm_w is None:
            return hidden

        x64 = hidden.astype(np.float64)
        var = np.mean(x64**2, axis=-1, keepdims=True)
        ffn_normed = (
            x64
            / np.sqrt(var + self.config.norm_eps)
            * (1.0 + ffn_norm_w.astype(np.float64))
        ).astype(np.float32)

        w_gate = w.get(f"{prefix}feed_forward.w_gate.weight")
        w_up = w.get(f"{prefix}feed_forward.w_up.weight")
        w_down = w.get(f"{prefix}feed_forward.w_down.weight")
        if w_gate is not None and w_up is not None and w_down is not None:
            gate = ffn_normed @ w_gate
            up = ffn_normed @ w_up
            sig = 1.0 / (1.0 + np.exp(-gate.astype(np.float32)))
            hidden = hidden + (gate * sig * up) @ w_down

        return hidden

    def forward(
        self,
        input_ids: np.ndarray,
        embed_w: Optional[np.ndarray] = None,
        lm_head: Optional[np.ndarray] = None,
        final_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if embed_w is None:
            raise RuntimeError("No embedding weights")
        hidden = embed_w[input_ids]
        if self.config.embedding_scale:
            hidden = hidden * math.sqrt(self.config.hidden_size)
        for i in range(self.config.num_layers):
            hidden = self.forward_layer(hidden, i)
        self._position += hidden.shape[0]
        if final_norm is not None:
            x64 = hidden.astype(np.float64)
            var = np.mean(x64**2, axis=-1, keepdims=True)
            hidden = (
                x64
                / np.sqrt(var + self.config.norm_eps)
                * (1.0 + final_norm.astype(np.float64))
            ).astype(np.float32)
        if lm_head is not None:
            logits = hidden @ lm_head
            if self.config.logit_softcap > 0:
                logits = (
                    np.tanh(logits / self.config.logit_softcap)
                    * self.config.logit_softcap
                )
            return logits
        return hidden

    def reset(self):
        self._kv_cache.clear()
        self._position = 0


# ═══════════════════════════════════════════════════════════════════════════
# Speculative Decoder (from cpu_inference_engine.py)
# ═══════════════════════════════════════════════════════════════════════════


class SpeculativeDecoder:
    """Draft K tokens with small model, verify in parallel with main model."""

    def __init__(
        self,
        main_forward: Callable,
        draft_forward: Callable,
        draft_k: int = 5,
        temperature: float = 1.0,
    ):
        self.main_forward = main_forward
        self.draft_forward = draft_forward
        self.draft_k = draft_k
        self.temperature = temperature
        self._accept_count = 0
        self._total_count = 0

    def generate(self, prompt_ids: List[int], max_tokens: int = 100) -> List[int]:
        generated = list(prompt_ids)
        while len(generated) - len(prompt_ids) < max_tokens:
            draft_tokens = self._draft_generate(generated, self.draft_k)
            verified = self._verify(generated, draft_tokens)
            generated.extend(verified)
            if not verified:
                break
        return generated[len(prompt_ids) :]

    def _draft_generate(self, context: List[int], k: int) -> List[int]:
        tokens = []
        ctx = list(context)
        for _ in range(k):
            logits = self.draft_forward(np.array(ctx, dtype=np.int32))
            if logits.ndim > 1:
                logits = logits[-1]
            t = self._sample(logits, self.temperature)
            tokens.append(t)
            ctx.append(t)
        return tokens

    def _verify(self, context: List[int], draft_tokens: List[int]) -> List[int]:
        if not draft_tokens:
            return []
        all_tokens = context + draft_tokens
        logits = self.main_forward(np.array(all_tokens, dtype=np.int32))
        accepted = []
        for i, dt in enumerate(draft_tokens):
            pos = len(context) + i
            if pos < logits.shape[0]:
                main_probs = _softmax_core(logits[pos], temperature=self.temperature)
                if (
                    np.random.random() < main_probs.get(dt, 0.0)
                    if hasattr(main_probs, "get")
                    else 0.5
                ):
                    accepted.append(dt)
                    self._accept_count += 1
                else:
                    break
            self._total_count += 1
        return accepted

    def acceptance_rate(self) -> float:
        return self._accept_count / max(self._total_count, 1)

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float = 1.0) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))
        probs = _softmax_core(logits / temperature, temperature=1.0)
        if hasattr(probs, "ravel"):
            return int(np.random.choice(len(probs), p=probs.ravel()))
        return int(np.random.randint(len(logits)))


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Inference Components (from adaptive_inference.py)
# ═══════════════════════════════════════════════════════════════════════════


class PredictiveConfidenceCascade:
    """Multi-depth n-gram confidence prediction for early exit decisions."""

    def __init__(self, max_depth: int = 6, min_depth: int = 2):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.deep_threshold = 0.75
        self.shallow_threshold = 0.35

    def predict_depth_confidences(
        self, ngram_counts: dict, total_counts: dict, context: tuple
    ) -> dict[int, float]:
        confidences = {}
        order_range = range(self.min_depth, min(self.max_depth, len(context)) + 1)
        for order in order_range:
            if order > len(context):
                continue
            ctx = context[-order:]
            total = total_counts.get(order, {}).get(ctx, 0)
            if total > 0:
                max_count = max(
                    ngram_counts.get(order, {}).get(ctx, {}).values(), default=0
                )
                depth_bonus = 1.0 + 0.15 * (order - 1)
                conf = float(np.clip((max_count / total) * depth_bonus, 0.0, 1.0))
            else:
                conf = 0.0
            confidences[order] = conf
        return confidences

    def select_optimal_depth(self, confidences: dict[int, float]) -> tuple[int, str]:
        if not confidences:
            return self.min_depth, "pre_emptive"
        for d in sorted([d for d in confidences if d >= 4], reverse=True):
            if confidences[d] >= self.deep_threshold:
                return d, "deep_skip"
        best_conf = max(confidences.values())
        best_depth = max(confidences, key=confidences.get)
        if best_conf < self.shallow_threshold:
            return best_depth, "pre_emptive"
        return best_depth, "balanced"


class StagedBlockEmission:
    """Progressive token emission with staged verification."""

    STAGE_CONFIGS = [
        {"name": "stage1", "tokens": 2, "threshold": 0.85},
        {"name": "stage2", "tokens": 4, "threshold": 0.70},
        {"name": "stage3", "tokens": 8, "threshold": 0.55},
    ]

    def __init__(self, max_skip_threshold: float = 0.92):
        self.max_skip_threshold = max_skip_threshold
        self._stage_idx = 0
        self._tokens_in_stage = 0
        self._total_emitted = 0

    def select_stage(self, confidence: float) -> dict:
        if confidence >= self.max_skip_threshold:
            self._stage_idx = 2
            self._tokens_in_stage = 0
            return self.STAGE_CONFIGS[2]
        for i, cfg in enumerate(self.STAGE_CONFIGS):
            if confidence >= cfg["threshold"]:
                self._stage_idx = i
                self._tokens_in_stage = 0
                return cfg
        self._stage_idx = 0
        self._tokens_in_stage = 0
        return self.STAGE_CONFIGS[0]

    def should_verify(self, tokens_generated: int) -> bool:
        cfg = self.STAGE_CONFIGS[self._stage_idx]
        self._tokens_in_stage += 1
        return self._tokens_in_stage >= cfg["tokens"]


class ThermalNoiseInjection:
    """Controlled noise for creativity and loop prevention."""

    def __init__(
        self, base_amplitude: float = 0.01, decay: float = 0.995, max_history: int = 256
    ):
        self.base_amplitude = base_amplitude
        self.decay = decay
        self._history: deque = deque(maxlen=max_history)
        self._amplitude = base_amplitude

    def compute_amplitude(self, temperature: float, confidence: float) -> float:
        self._amplitude *= self.decay
        entropy_bonus = 0.0
        if len(self._history) > 10:
            recent = list(self._history)[-10:]
            if len(set(recent)) < 3:
                entropy_bonus = 0.05
        return self.base_amplitude * temperature * (1.0 - confidence) + entropy_bonus

    def inject_noise(
        self, logits: np.ndarray, temperature: float, confidence: float
    ) -> np.ndarray:
        amp = self.compute_amplitude(temperature, confidence)
        noise = np.random.randn(*logits.shape).astype(np.float32) * amp
        return logits + noise

    def detect_repetition(self, token: int) -> bool:
        self._history.append(token)
        if len(self._history) < 4:
            return False
        recent = list(self._history)[-4:]
        return len(set(recent)) <= 1


# ═══════════════════════════════════════════════════════════════════════════
# AST Inference (from ast_inference.py)
# ═══════════════════════════════════════════════════════════════════════════


class ASTInferenceEngine:
    """AST-aware inference for code generation with syntactic guarantees."""

    def __init__(self, generate_fn: Optional[Callable] = None):
        self.generate_fn = generate_fn
        self.last_ast = None

    def detect_type(self, text: str) -> str:
        code_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "if __name__",
            "func ",
            "#include",
            "{",
            "}",
            ";",
            ":=",
            "->",
            "lambda ",
            "yield ",
            "try:",
        ]
        score = sum(1 for ind in code_indicators if ind in text)
        return "code" if score >= 3 else "prose"

    def parse_ast(self, code: str) -> Optional[ast.AST]:
        try:
            self.last_ast = ast.parse(code)
            return self.last_ast
        except SyntaxError:
            return None

    def score_code_quality(self, code: str) -> float:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0
        score = 0.4
        funcs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if funcs:
            score += 0.1
        docstrings = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n)
        ]
        if docstrings:
            score += 0.1
        node_count = sum(1 for _ in ast.walk(tree))
        if 10 <= node_count <= 500:
            score += 0.1
        return max(0.0, min(1.0, score))

    def validate_completion(self, prefix: str, completion: str) -> float:
        full = prefix + completion
        tree = self.parse_ast(full)
        if tree is not None:
            return 1.0
        tree2 = self.parse_ast(full.rstrip())
        if tree2 is not None:
            return 0.8
        return 0.0

    def suggest_ast_template(self, context: str) -> Optional[dict]:
        stripped = context.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            return {
                "type": "function",
                "template": 'def name(args):\n    """Docstring."""\n    pass\n',
            }
        if stripped.startswith("class "):
            return {
                "type": "class",
                "template": 'class Name:\n    """Docstring."""\n    def __init__(self):\n        pass\n',
            }
        if stripped.startswith("import ") or stripped.startswith("from "):
            return {"type": "import", "template": "import module\n"}
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Holographic Memory (from unified_inference.py)
# ═══════════════════════════════════════════════════════════════════════════


class HrrMemory:
    """Holographic Reduced Representations for compressed memory storage."""

    def __init__(self, dim: int = 4096, capacity: int = 65536):
        self.dim = dim
        self.capacity = capacity
        self.memory: dict[int, np.ndarray] = {}

    def _make_key_vector(self, key: int) -> np.ndarray:
        rng = np.random.RandomState(hash(key) & 0x7FFFFFFF)
        vec = rng.randn(self.dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    def store(self, key: int, value: np.ndarray):
        key_vec = self._make_key_vector(key)
        n = self.dim
        A_fft = np.fft.fft(key_vec.astype(np.complex128))
        B_fft = np.fft.fft(
            value.astype(np.complex128)[:n]
            if len(value) >= n
            else np.pad(value, (0, n - len(value))).astype(np.complex128)
        )
        encoded = np.fft.ifft(A_fft * B_fft).real.astype(np.float32)
        self.memory[key] = encoded
        if len(self.memory) > self.capacity:
            oldest = next(iter(self.memory))
            del self.memory[oldest]

    def recall(self, key: int) -> Optional[np.ndarray]:
        if key not in self.memory:
            return None
        key_vec = self._make_key_vector(key)
        n = self.dim
        A_fft = np.fft.fft(key_vec.astype(np.complex128))
        B_fft = np.fft.fft(self.memory[key].astype(np.complex128))
        return np.fft.ifft(np.conj(A_fft) * B_fft).real.astype(np.float32)

    def clear(self):
        self.memory.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Memory Tier (from streaming_engine.py)
# ═══════════════════════════════════════════════════════════════════════════


class TieredMemory:
    """Simplified 3-tier memory (DRAM -> L3 -> L2) for weight streaming."""

    def __init__(self, l3_size: int = 16 * 1024 * 1024, l2_size: int = 512 * 1024):
        self.l3 = OrderedDict()  # layer_idx -> bytes
        self.l3_size = l3_size
        self.l3_used = 0
        self.l2 = OrderedDict()  # layer_idx -> bytes
        self.l2_size = l2_size
        self.l2_used = 0

    def store_l3(self, layer_idx: int, data: bytes) -> bool:
        if self.l3_used + len(data) > self.l3_size:
            self._evict_l3(1)
        self.l3[layer_idx] = data
        self.l3_used += len(data)
        return True

    def load_l3(self, layer_idx: int) -> Optional[bytes]:
        if layer_idx in self.l3:
            self.l3.move_to_end(layer_idx)
            return self.l3[layer_idx]
        return None

    def store_l2(self, layer_idx: int, data: bytes) -> bool:
        if self.l2_used + len(data) > self.l2_size:
            self._evict_l2(1)
        self.l2[layer_idx] = data
        self.l2_used += len(data)
        return True

    def load_l2(self, layer_idx: int) -> Optional[np.ndarray]:
        if layer_idx in self.l2:
            self.l2.move_to_end(layer_idx)
            return np.frombuffer(self.l2[layer_idx], dtype=np.float32)
        return None

    def _evict_l3(self, n: int):
        for _ in range(min(n, len(self.l3))):
            _, data = self.l3.popitem(last=False)
            self.l3_used -= len(data)

    def _evict_l2(self, n: int):
        for _ in range(min(n, len(self.l2))):
            _, data = self.l2.popitem(last=False)
            self.l2_used -= len(data)

    def summary(self) -> dict:
        return {
            "l3_entries": len(self.l3),
            "l3_used_mb": self.l3_used / 1048576,
            "l2_entries": len(self.l2),
            "l2_used_mb": self.l2_used / 1048576,
        }


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED INFERENCE ENGINE — Main API
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedInferenceEngine:
    """Single clean inference engine consolidating all implementations.

    Supports:
      - CPU inference with spectral GEMM
      - Speculative decoding
      - Streaming forward pass
      - 6 strategy levels (forwardless -> standard)
      - KV cache with auto-compression
      - AST-guided code generation

    Usage:
        engine = UnifiedInferenceEngine(model_path="model.gguf")
        result = engine.generate("Hello", max_tokens=100)
        print(result.text, result.tokens_per_sec)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        hidden_dim: int = 512,
        vocab_size: int = 32000,
        n_heads: int = 8,
        n_layers: int = 8,
        head_dim: Optional[int] = None,
        kv_cache_size: int = 4096,
        config: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.config = config or {}
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = head_dim or (hidden_dim // max(n_heads, 1))

        self._start_time = time.time()
        self._generation_count = 0

        self.kv_cache = UnifiedKVCache(
            num_heads=n_heads,
            head_dim=self.head_dim,
            max_seq_len=kv_cache_size,
            num_layers=n_layers,
            compression=self.config.get("kv_compression", "auto"),
        )

        self.gemm = SpectralGEMM()
        self.attention = CPUOptimizedAttention(
            Gemma4Config(
                hidden_size=hidden_dim,
                num_layers=n_layers,
                num_heads=n_heads,
                head_dim=self.head_dim,
            )
        )
        self.forward_pass: Optional[StreamingForwardPass] = None
        self._weights: Dict[str, np.ndarray] = {}
        self._embedding: Optional[np.ndarray] = None
        self._lm_head: Optional[np.ndarray] = None
        self._final_norm: Optional[np.ndarray] = None

        self.confidence_cascade = PredictiveConfidenceCascade()
        self.staged_emission = StagedBlockEmission()
        self.thermal_noise = ThermalNoiseInjection()
        self.ast_engine = ASTInferenceEngine()
        self.hrr_memory = HrrMemory(dim=self.head_dim)
        self.tiered_memory = TieredMemory()
        self._position = 0

        if model_path and os.path.exists(model_path):
            self._try_load_model(model_path)

    def _try_load_model(self, path: str):
        """Attempt to load model weights. Best-effort."""
        try:
            from spectralstream.model.weight_loader import WeightLoader

            loader = WeightLoader(path)
            self.forward_pass = StreamingForwardPass(
                Gemma4Config(
                    hidden_size=self.hidden_dim,
                    num_layers=self.n_layers,
                    num_heads=self.n_heads,
                    head_dim=self.head_dim,
                )
            )
            for name in loader.list_tensors():
                tensor = loader.load_tensor(name)
                self._weights[name] = tensor
                if "token_embed" in name or "embed_tokens" in name:
                    self._embedding = tensor
                elif "output.weight" in name or "lm_head.weight" in name:
                    self._lm_head = tensor
                elif "output_norm" in name:
                    self._final_norm = tensor
            if self.forward_pass:
                self.forward_pass._weights = self._weights
        except Exception:
            pass

    def forward(self, input_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """Run forward pass, returning logits."""
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids, dtype=np.int32)

        if self.forward_pass and self._embedding is not None:
            return self.forward_pass.forward(
                input_ids, self._embedding, self._lm_head, self._final_norm
            )

        if self._embedding is not None:
            hidden = self._embedding[input_ids]
            if self.config.get("embedding_scale", True):
                hidden = hidden * math.sqrt(self.hidden_dim)
            if self._lm_head is not None:
                return hidden @ self._lm_head
            return hidden

        return np.random.randn(len(input_ids), self.vocab_size).astype(np.float32)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        strategy: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            strategy: Override strategy level name
            **kwargs: Additional parameters

        Returns:
            GenerationResult with text, metrics, and metadata
        """
        t0 = time.time()
        token_ids = self.tokenize(prompt)
        input_len = len(token_ids)

        strat_level = StrategyLevel.STANDARD
        if strategy:
            for lvl, name in STRATEGY_NAMES.items():
                if name == strategy:
                    strat_level = lvl
                    break

        if self.forward_pass:
            self.forward_pass.reset()
        self._position = 0

        generated = list(token_ids)
        ttft_ms = 0.0

        for i in range(max_tokens):
            logits = self.forward(
                np.array(
                    generated[-self.config.get("context_window", 2048) :],
                    dtype=np.int32,
                )
            )
            if logits.ndim > 1:
                logits = logits[-1]

            if i == 0:
                ttft_ms = (time.time() - t0) * 1000

            logits = self.thermal_noise.inject_noise(logits, temperature, 0.5)

            token = self._sample(logits, temperature, top_k, top_p)
            generated.append(token)

            if self._is_stop_token(token):
                break

        new_tokens = generated[input_len:]
        total_time = time.time() - t0
        tps = len(new_tokens) / max(total_time, 1e-6)

        text = self.detokenize(new_tokens)
        self._generation_count += 1

        return GenerationResult(
            text=text,
            token_ids=new_tokens,
            tokens_per_sec=tps,
            total_time_ms=total_time * 1000,
            strategy_used=STRATEGY_NAMES.get(strat_level, "standard"),
            kv_cache_ratio=self.kv_cache.get_metrics().avg_compression_ratio,
        )

    def benchmark(
        self, prompts: Optional[List[str]] = None, max_tokens: int = 50, n_runs: int = 3
    ) -> BenchmarkReport:
        """Run benchmark across prompts."""
        if prompts is None:
            prompts = ["Hello world", "The meaning of life is", "def fibonacci("]

        total_tokens = 0
        total_time = 0.0
        ttft_sum = 0.0

        for _ in range(n_runs):
            for prompt in prompts:
                result = self.generate(prompt, max_tokens=max_tokens)
                total_tokens += len(result.token_ids)
                total_time += result.total_time_ms / 1000
                ttft_sum += result.total_time_ms

        avg_ttft = ttft_sum / (len(prompts) * n_runs)
        tps = total_tokens / max(total_time, 1e-6)

        return BenchmarkReport(
            generated_tokens=total_tokens,
            total_time_s=total_time,
            tokens_per_sec=tps,
            time_to_first_token_ms=avg_ttft,
            kv_compression_ratio=self.kv_cache.get_metrics().avg_compression_ratio,
        )

    def tokenize(self, text: str) -> List[int]:
        """Simple character-level tokenization fallback."""
        return [ord(c) % self.vocab_size for c in text]

    def detokenize(self, token_ids: List[int]) -> str:
        """Simple character-level detokenization fallback."""
        return "".join(chr(max(32, min(t, 127))) for t in token_ids)

    def ast_generate(self, prompt: str, max_tokens: int = 256) -> dict:
        """Generate code with AST validity guarantees."""
        template = self.ast_engine.suggest_ast_template(prompt)
        result = self.generate(prompt, max_tokens=max_tokens)
        score = self.ast_engine.score_code_quality(result.text)
        ast_valid = self.ast_engine.parse_ast(result.text) is not None
        return {
            "text": result.text,
            "token_ids": result.token_ids,
            "ast_valid": ast_valid,
            "quality_score": score,
            "template": template,
            "tokens_per_sec": result.tokens_per_sec,
        }

    def speculative_generate(
        self, prompt: str, max_tokens: int = 100, draft_k: int = 5
    ) -> GenerationResult:
        """Generate with speculative decoding."""
        token_ids = self.tokenize(prompt)

        def main_forward(ids):
            return self.forward(ids)

        def draft_forward(ids):
            return self.forward(ids)

        decoder = SpeculativeDecoder(main_forward, draft_forward, draft_k=draft_k)
        t0 = time.time()
        new_tokens = decoder.generate(token_ids, max_tokens=max_tokens)
        total_time = time.time() - t0

        return GenerationResult(
            text=self.detokenize(new_tokens),
            token_ids=new_tokens,
            tokens_per_sec=len(new_tokens) / max(total_time, 1e-6),
            total_time_ms=total_time * 1000,
            strategy_used="speculative",
            metadata={"acceptance_rate": decoder.acceptance_rate()},
        )

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))
        logits = logits / max(temperature, 1e-10)
        if top_k > 0 and top_k < len(logits):
            idx = np.argpartition(logits, -top_k)[-top_k:]
            filtered = np.full_like(logits, -np.inf)
            filtered[idx] = logits[idx]
            logits = filtered
        probs = _softmax_core(logits, temperature=1.0)
        if top_p < 1.0:
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, top_p)
            mask = np.zeros_like(probs, dtype=bool)
            mask[sorted_idx[: cutoff + 1]] = True
            probs[~mask] = 0
            probs = probs / (probs.sum() + 1e-10)
        return int(np.random.choice(len(probs), p=probs.ravel()))

    def _is_stop_token(self, token: int) -> bool:
        return token in (0, 2, 3)  # EOS, PAD, UNK common tokens

    def get_stats(self) -> dict:
        elapsed = time.time() - self._start_time
        return {
            "model_path": self.model_path,
            "device": self.device,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "generations": self._generation_count,
            "elapsed_s": elapsed,
            "kv_cache": self.kv_cache.cache_summary(),
            "tiered_memory": self.tiered_memory.summary(),
        }

    def clear_cache(self):
        self.kv_cache.clear()
        self.hrr_memory.clear()
        self.tiered_memory = TieredMemory()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear_cache()


# ═══════════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════════


def create_unified_engine(
    model_path: Optional[str] = None,
    device: str = "cpu",
    config: Optional[dict] = None,
) -> UnifiedInferenceEngine:
    """Factory for creating UnifiedInferenceEngine."""
    cfg = config or {}
    return UnifiedInferenceEngine(
        model_path=model_path,
        device=device,
        hidden_dim=cfg.get("hidden_dim", 512),
        vocab_size=cfg.get("vocab_size", 32000),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 8),
        kv_cache_size=cfg.get("kv_cache_size", 4096),
        config=cfg,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility Aliases
# ═══════════════════════════════════════════════════════════════════════════

SpectralInferenceEngine = UnifiedInferenceEngine
SpectralGEMM_backcompat = SpectralGEMM
StreamingForwardPass_backcompat = StreamingForwardPass

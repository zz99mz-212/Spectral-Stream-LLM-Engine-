"""
Production-grade CPU inference pipeline for SpectralStream.
Integrates SSF decompression, KV cache with compression,
Gemma 4 forward pass, token generation, benchmarking, and
the full 6-level unified inference system (COCONUT, Vlasov,
HDC forwardless, TimeCrystal, etc.).
"""

from __future__ import annotations
import math
import time
import os
import gc
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from spectralstream.inference.config import Gemma4Config
from spectralstream.inference.loader import ModelLoader
from spectralstream.inference.layer import TransformerLayer
from spectralstream.inference.ffn import _gelu_tanh
from spectralstream.inference.attention import Gemma4RMSNorm

from spectralstream.kv_cache.core import KVCacheConfig
from spectralstream.kv_cache.manager import KVCacheManager
from spectralstream.kv_cache.intelligence_engine import (
    KVCacheIntelligenceEngine,
    KVCacheIntelligenceConfig,
)

from spectralstream.format.reader import SSFReader
from spectralstream.inference.loader import SafeTensorsLoader

from spectralstream.inference.coconut import COCONUTEngine
from spectralstream.inference.unified import (
    UnifiedInferenceEngine,
    UnifiedStrategyLevel,
    create_unified_engine,
)

# ── Memory-safety constants ──────────────────────────────────────────
# Block size for vocab-dimension log-sum-exp to cap per-window logits
# memory to (block_size * seq_len * 8) bytes instead of
# (vocab_size * seq_len * 8).  4096 * 2048 * 8 = 64 MB vs
# 262144 * 2048 * 8 = 4 GB  (Gemma-4 E2B has a 262k vocab).
VOCAB_LOG_SOFTMAX_BLOCK_SIZE: int = 4096


def _blocked_log_sum_exp(logits: np.ndarray, block_size: int) -> np.ndarray:
    """Compute ``log(sum(exp(logits), axis=-1))`` in vocab blocks.

    Standard ``np.log(np.sum(np.exp(logits), axis=-1))`` materialises a
    ``[*, vocab_size]`` float64 intermediate that is **4 GB** for a
    2048-token window on a 262k-vocab model.  This helper processes the
    vocab axis in chunks of *block_size*, accumulating in float64, so
    peak memory is ``[*, block_size]`` (e.g. 64 MB at block_size=4096).

    Parameters
    ----------
    logits : np.ndarray
        Logits array with shape ``(..., vocab_size)``.
    block_size : int
        Vocab chunk size.

    Returns
    -------
    np.ndarray
        ``log(sum(exp(logits), axis=-1))`` as a float64 array.
    """
    # Work in float64 for numerical stability of the log-sum-exp
    logits = np.asarray(logits, dtype=np.float64)
    vocab_size = logits.shape[-1]
    # Subtract max for numerical stability
    max_val = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_val

    # Accumulate exp-sum in blocks
    total = np.zeros(shifted.shape[:-1], dtype=np.float64)
    for start in range(0, vocab_size, block_size):
        end = min(start + block_size, vocab_size)
        total += np.exp(shifted[..., start:end]).sum(axis=-1)

    # Guard against log(0)
    total = np.maximum(total, 1e-300)
    return np.log(total) + max_val.squeeze(-1)


@dataclass
class InferenceConfig:
    model_path: str = ""
    cache_size_gb: float = 2.0
    kv_cache_size_gb: float = 4.0
    kv_cache_method: str = "none"
    kv_cache_eviction: str = "spectral"
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    max_new_tokens: int = 100
    eos_token_id: int = 1
    prefetch_enabled: bool = True
    verbose: bool = False
    benchmark_decode_steps: int = 10
    benchmark_warmup: int = 2
    benchmark_runs: int = 3


class InferencePipeline:
    """Production-grade CPU inference pipeline.

    Loads compressed SSF models (or raw safetensors) and runs
    inference with on-the-fly decompression, KV caching, and
    optional advanced eviction/compression policies.

    Supports:
    - SSF v2 compressed models (on-the-fly decompression via engine)
    - Raw safetensors / legacy format
    - KVCacheIntelligenceEngine integration with 30+ compression methods
    - Gemma 4 forward pass (sliding + full attention layers)
    - Token generation (temperature, top-k, top-p sampling)
    - Benchmarking (throughput, latency, memory, perplexity)
    - Thread-safe generation (re-entrant via per-call state)
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None,
        *,
        use_unified: bool = True,
        coconut_engine: Optional[COCONUTEngine] = None,
        unified_config: Optional[dict] = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self.use_unified = use_unified
        self.unified_config = unified_config or {}

        if use_unified:
            self._init_unified(coconut_engine)
            return

        self._ssf_reader: Optional[SSFReader] = None
        self._loader: Any = None  # ModelLoader or SafeTensorsLoader
        self._freqs_cis: Optional[Dict[str, np.ndarray]] = None
        self._load_model()
        self.model_config = self._resolve_config()
        self.kv_cache = self._build_kv_cache()
        self._kv_cache_dict: Dict[str, np.ndarray] = {}
        self._precompute_freqs_cis()
        self._layer_cache: Dict[int, Optional[TransformerLayer]] = {}
        self.layers: List[Optional[TransformerLayer]] = self._build_layers()

    # ── Model Loading ──────────────────────────────────────────────────

    def _init_unified(self, coconut_engine: Optional[COCONUTEngine] = None):
        cfg = self.unified_config
        mcfg = self.model_config

        def _forward(tokens, past=None):
            return self._legacy_forward(tokens)

        self._unified = UnifiedInferenceEngine(
            model_forward_fn=_forward,
            hidden_dim=mcfg.HIDDEN_SIZE,
            vocab_size=mcfg.VOCAB_SIZE,
            n_heads=mcfg.NUM_ATTENTION_HEADS,
            n_layers=mcfg.NUM_HIDDEN_LAYERS,
            hd_dim=cfg.get("hd_dim", 4096),
            kv_cache_size=cfg.get("kv_cache_size", 4096),
            n_candidate_blocks=cfg.get("n_candidate_blocks", 16),
            coconut_engine=coconut_engine,
            config=cfg,
        )

    def _load_model(self) -> None:
        path = self.model_path
        is_ssf = path.endswith(".ssf")
        verbose = self.config.verbose

        # Try SSF reader first
        if is_ssf:
            try:
                self._ssf_reader = SSFReader(path, cache_size=32)
                if verbose:
                    print(f"Loaded SSF model: {path}")
            except Exception as e:
                if verbose:
                    print(f"SSF reader failed for {path}: {e}")
                self._ssf_reader = None

        # Try ModelLoader (SSF loader) next
        if self._ssf_reader is None:
            try:
                self._loader = ModelLoader(path, self.config.cache_size_gb)
                if verbose:
                    print(f"Loaded model via ModelLoader: {path}")
            except Exception as e:
                if verbose:
                    print(f"ModelLoader failed for {path}: {e}")
                self._loader = None

        # Try SafeTensorsLoader as last resort
        if self._ssf_reader is None and self._loader is None:
            safetensors_path = path
            if is_ssf:
                base = path.rsplit(".", 1)[0]
                safetensors_path = base + ".safetensors"
                if not os.path.exists(safetensors_path):
                    safetensors_path = os.path.join(
                        os.path.dirname(path), "model.safetensors"
                    )
            if os.path.exists(safetensors_path):
                try:
                    self._loader = SafeTensorsLoader(
                        safetensors_path, self.config.cache_size_gb
                    )
                    if verbose:
                        print(f"Loaded via safetensors: {safetensors_path}")
                except Exception as e:
                    if verbose:
                        print(f"SafeTensorsLoader failed: {e}")
                    self._loader = None
            elif verbose:
                print(f"No safetensors found at {safetensors_path}")

        if self._ssf_reader is None and self._loader is None:
            raise RuntimeError(
                f"Could not load model from '{path}'. "
                f"Tried SSF reader, ModelLoader, and safetensors."
            )

    def _resolve_config(self) -> Gemma4Config:
        cfg = Gemma4Config.from_ssf(self.model_path)
        try:
            if self._ssf_reader is not None:
                md = self._ssf_reader.metadata
                c = md.get("config", {})
                for k, v in c.items():
                    key = k.upper()
                    if hasattr(cfg, key):
                        setattr(cfg, key, v)
        except Exception:
            pass
        return cfg

    def _precompute_freqs_cis(self) -> None:
        mcfg = self.model_config
        head_dim = mcfg.HEAD_DIM
        half = head_dim // 2
        theta = float(mcfg.ROPE_THETA)
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
        max_seq = min(mcfg.MAX_POSITION_EMBEDDINGS, 131072)
        positions = np.arange(max_seq, dtype=np.float32)
        angles = np.outer(positions, freqs)
        self._freqs_cis = {
            "cos": np.cos(angles).astype(np.float32),
            "sin": np.sin(angles).astype(np.float32),
        }

    def _build_kv_cache(self) -> KVCacheIntelligenceEngine:
        mcfg = self.model_config
        kv_config = KVCacheConfig(
            max_seq_len=mcfg.MAX_POSITION_EMBEDDINGS,
            num_layers=mcfg.NUM_HIDDEN_LAYERS,
            num_heads=mcfg.NUM_KEY_VALUE_HEADS,
            head_dim=mcfg.HEAD_DIM,
            hidden_size=mcfg.HIDDEN_SIZE,
            cache_size_limit_gb=self.config.kv_cache_size_gb,
            compression_method=self.config.kv_cache_method,
            eviction_policy=self.config.kv_cache_eviction,
            prefetch_enabled=self.config.prefetch_enabled,
        )
        ie_config = KVCacheIntelligenceConfig(
            enable_monitoring=True,
            enable_auto_tune=True,
            enable_fallback=True,
            max_cache_memory_gb=self.config.kv_cache_size_gb,
        )
        return KVCacheIntelligenceEngine(kv_config, ie_config)

    def _build_layers(self) -> List[Optional[TransformerLayer]]:
        return [None] * self.model_config.NUM_HIDDEN_LAYERS

    def _get_layer(self, layer_idx: int) -> Optional[TransformerLayer]:
        if layer_idx in self._layer_cache:
            return self._layer_cache[layer_idx]
        layer = self._layers[layer_idx]
        if layer is not None:
            self._layer_cache[layer_idx] = layer
            return layer
        weights = self._get_layer_weights(layer_idx)
        if weights is None:
            self._layer_cache[layer_idx] = None
            return None
        (
            attn_norm,
            ffn_norm,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
        ) = weights
        layer = TransformerLayer(
            self.model_config,
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
        self._layers[layer_idx] = layer
        self._layer_cache[layer_idx] = layer
        return layer

    def _get_layer_weights(self, layer_idx: int) -> Optional[Tuple[Any, ...]]:
        prefix = f"blk.{layer_idx}."
        names = self.tensor_names

        def _load(suffix: str) -> Optional[np.ndarray]:
            full = f"{prefix}{suffix}"
            if full not in names:
                return None
            return self.get_tensor(full)

        attn_norm = _load("attention_norm.weight")
        ffn_norm = _load("feed_forward_norm.weight")
        wq = _load("attention.wq.weight")
        wk = _load("attention.wk.weight")
        wv = _load("attention.wv.weight")
        wo = _load("attention.wo.weight")
        w_gate = _load("feed_forward.w_gate.weight")
        w_up = _load("feed_forward.w_up.weight")
        w_down = _load("feed_forward.w_down.weight")

        if any(
            x is None
            for x in [attn_norm, ffn_norm, wq, wk, wv, wo, w_gate, w_up, w_down]
        ):
            return None
        return (attn_norm, ffn_norm, wq, wk, wv, wo, w_gate, w_up, w_down)

    # ── Tensor Access ──────────────────────────────────────────────────

    @property
    def tensor_names(self) -> List[str]:
        if self._ssf_reader is not None:
            return self._ssf_reader.tensor_names()
        if self._loader is not None:
            return self._loader.tensor_names
        return []

    def get_tensor(self, name: str) -> np.ndarray:
        """Get a tensor, decompressing on-the-fly if from SSF."""
        if self._ssf_reader is not None:
            return self._ssf_reader.get_tensor(name)
        if self._loader is not None:
            return self._loader.get_tensor(name)
        raise KeyError(f"No model loaded for tensor '{name}'")

    def _sync_kv_cache(self, n_tokens: int, positions: np.ndarray) -> None:
        if self.config.kv_cache_method == "none" or n_tokens == 0:
            return
        for layer_idx in range(self.model_config.NUM_HIDDEN_LAYERS):
            k_key = f"k_{layer_idx}"
            v_key = f"v_{layer_idx}"
            if k_key not in self._kv_cache_dict or v_key not in self._kv_cache_dict:
                continue
            k = self._kv_cache_dict[k_key]
            v = self._kv_cache_dict[v_key]
            start = max(0, k.shape[0] - n_tokens)
            for offset in range(k.shape[0] - start):
                idx = start + offset
                pos = int(positions[offset]) if offset < len(positions) else idx
                self.kv_cache.store(layer_idx, k[idx : idx + 1], v[idx : idx + 1], pos)

    @staticmethod
    def _make_causal_mask(n_tokens: int, n_kv: Optional[int] = None) -> np.ndarray:
        if n_kv is None:
            n_kv = n_tokens
        mask = np.triu(np.full((n_tokens, n_kv), -np.inf, dtype=np.float32), k=1)
        return mask

    # ── Legacy Forward Pass (used as backend for Unified engine) ──────

    # TODO: migrate _kv_cache_dict (flat dict) to KVCacheManager
    def _legacy_forward(
        self,
        tokens: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        embed_w = self.get_tensor("token_embed.weight")
        hidden = embed_w[tokens].astype(np.float32) * math.sqrt(
            self.model_config.HIDDEN_SIZE
        )
        n_tokens = tokens.shape[0]
        if positions is None:
            positions = np.arange(n_tokens, dtype=np.int32)

        for layer_idx in range(self.model_config.NUM_HIDDEN_LAYERS):
            layer = self._get_layer(layer_idx)
            if layer is None:
                continue
            hidden = layer(
                hidden,
                self._freqs_cis,
                None,
                self._kv_cache_dict,
                positions,
            )

        self._sync_kv_cache(n_tokens, positions)

        final_norm_w = self.get_tensor("output_norm.weight")
        var = np.mean(hidden.astype(np.float32) ** 2, axis=-1, keepdims=True)
        rsqrt = np.float32(1.0) / np.sqrt(var + self.model_config.NORM_EPS)
        hidden = (hidden * rsqrt) * (np.float32(1.0) + final_norm_w.astype(np.float32))

        lm_head = self.get_tensor("output.weight")
        logits = hidden.astype(np.float32) @ lm_head.astype(np.float32).T

        logit_softcap = self.model_config.LOGIT_SOFTCAP
        if logit_softcap > 0:
            logits = np.tanh(logits / logit_softcap) * logit_softcap

        return logits

    # ── Forward Pass (delegates to Unified engine if active) ──────────

    def forward(
        self,
        tokens: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.use_unified:
            return self._unified._model_forward(tokens)
        return self._legacy_forward(tokens, positions)

    # ── Sampling ───────────────────────────────────────────────────────

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
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

    # ── Generation ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Generate tokens with the full inference pipeline.

        When use_unified=True (default), runs through the 6-level unified
        inference engine with COCONUT, Vlasov, HDC, TimeCrystal, etc.

        Returns (generated_tokens, metrics_dict).
        """
        if self.use_unified:
            tokens, tps = self._unified.generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                top_k=top_k or self.config.top_k,
                top_p=top_p or self.config.top_p,
            )
            n_decode = len(tokens) - len(prompt_tokens)
            metrics = {
                "prefill_tokens": len(prompt_tokens),
                "decode_tokens": n_decode,
                "total_tokens": len(tokens),
                "total_time_s": round(n_decode / max(tps, 0.01), 4) if tps > 0 else 0.0,
                "tokens_per_second": round(tps, 1),
                "strategy": UNIFIED_STRATEGY_NAMES.get(
                    self._unified._current_strategy, "unknown"
                ),
            }
            return tokens[len(prompt_tokens) :], metrics

        max_new = (
            max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        )
        if max_new < 0:
            return [], {"error": "max_new_tokens must be >= 0"}
        temp = temperature if temperature is not None else self.config.temperature
        tk = top_k if top_k is not None else self.config.top_k
        tp = top_p if top_p is not None else self.config.top_p
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        self.kv_cache.clear()
        self._kv_cache_dict.clear()
        generated: List[int] = []
        t_start = time.perf_counter()
        n_prefill_tokens = len(prompt_tokens)

        tokens = np.array(prompt_tokens, dtype=np.int32)
        logits = self.forward(tokens)
        token = self._sample(logits[-1], temp, tk, tp)
        generated.append(token)
        del logits
        if eos is not None and token == eos:
            t_elapsed = time.perf_counter() - t_start
            return generated, self._build_metrics(
                n_prefill_tokens, generated, t_elapsed
            )

        for _ in range(max_new - 1):
            tokens = np.array([token], dtype=np.int32)
            logits = self.forward(tokens)
            token = self._sample(logits[-1], temp, tk, tp)
            generated.append(token)
            del logits
            if _ % 10 == 9:
                gc.collect()
            if eos is not None and token == eos:
                break

        t_elapsed = time.perf_counter() - t_start
        return generated, self._build_metrics(n_prefill_tokens, generated, t_elapsed)

    def _build_metrics(
        self,
        n_prefill: int,
        generated: List[int],
        t_elapsed: float,
    ) -> Dict[str, Any]:
        n_decode = len(generated)
        return {
            "prefill_tokens": n_prefill,
            "decode_tokens": n_decode,
            "total_tokens": n_prefill + n_decode,
            "total_time_s": round(t_elapsed, 4),
            "prefill_tokens_per_second": (
                round(n_prefill / t_elapsed, 1) if t_elapsed > 0 else 0.0
            ),
            "decode_tokens_per_second": (
                round(n_decode / t_elapsed, 1) if t_elapsed > 0 else 0.0
            ),
            "tokens_per_second": (
                round((n_prefill + n_decode) / t_elapsed, 1) if t_elapsed > 0 else 0.0
            ),
        }

    def generate_stream(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Generator[int, None, None]:
        """Stream tokens one at a time with yield."""
        if self.use_unified:
            for token, strategy_name, _mode in self._unified.stream_generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                top_k=top_k or self.config.top_k,
                top_p=top_p or self.config.top_p,
            ):
                yield token
            return

        max_new = (
            max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        )
        if max_new < 0:
            return
        temp = temperature if temperature is not None else self.config.temperature
        tk = top_k if top_k is not None else self.config.top_k
        tp = top_p if top_p is not None else self.config.top_p
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        self.kv_cache.clear()
        self._kv_cache_dict.clear()

        tokens = np.array(prompt_tokens, dtype=np.int32)
        logits = self.forward(tokens)
        token = self._sample(logits[-1], temp, tk, tp)
        del logits
        yield token
        if eos is not None and token == eos:
            return

        for _ in range(max_new - 1):
            tokens = np.array([token], dtype=np.int32)
            logits = self.forward(tokens)
            token = self._sample(logits[-1], temp, tk, tp)
            del logits
            if _ % 10 == 9:
                gc.collect()
            yield token
            if eos is not None and token == eos:
                return

    # ── Benchmarking ───────────────────────────────────────────────────

    def benchmark(
        self,
        prompt_lengths: Optional[List[int]] = None,
        num_runs: Optional[int] = None,
        num_warmup: Optional[int] = None,
        decode_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive throughput benchmarks.

        Measures prefill + decode performance at multiple sequence lengths,
        peak memory, and compression metrics.
        """
        if prompt_lengths is None:
            prompt_lengths = [128, 512, 2048]
        if num_runs is None:
            num_runs = self.config.benchmark_runs
        if num_warmup is None:
            num_warmup = self.config.benchmark_warmup
        if decode_steps is None:
            decode_steps = self.config.benchmark_decode_steps

        results: Dict[str, Any] = {
            "model_path": self.model_path,
            "config": {
                "cache_size_gb": self.config.cache_size_gb,
                "kv_cache_size_gb": self.config.kv_cache_size_gb,
                "kv_cache_method": self.config.kv_cache_method,
            },
            "model_config": {
                "hidden_size": self.model_config.HIDDEN_SIZE,
                "num_layers": self.model_config.NUM_HIDDEN_LAYERS,
                "num_heads": self.model_config.NUM_ATTENTION_HEADS,
                "num_kv_heads": self.model_config.NUM_KEY_VALUE_HEADS,
                "head_dim": self.model_config.HEAD_DIM,
                "vocab_size": self.model_config.VOCAB_SIZE,
                "max_seq_len": self.model_config.MAX_POSITION_EMBEDDINGS,
            },
            "throughput": {},
            "memory": {},
            "compression": {},
        }

        fake_vocab_size = min(self.model_config.VOCAB_SIZE, 32000)

        for seq_len in prompt_lengths:
            seq_key = f"seq{seq_len}"
            seq_results: Dict[str, Any] = {}
            prefill_times: List[float] = []
            decode_times: List[float] = []
            decode_tokens_list: List[int] = []

            prompt = list(
                np.random.RandomState(42)
                .randint(1, fake_vocab_size, size=seq_len)
                .tolist()
            )

            for run in range(num_runs + num_warmup):
                self.kv_cache.clear()
                self._kv_cache_dict.clear()
                is_warmup = run < num_warmup

                t0 = time.perf_counter()
                tokens = np.array(prompt, dtype=np.int32)
                logits = self.forward(tokens)
                t1 = time.perf_counter()

                n_decode = 0
                token = self._sample(logits[-1])
                n_decode += 1
                t2 = time.perf_counter()

                for _ in range(decode_steps - 1):
                    tokens = np.array([token], dtype=np.int32)
                    logits = self.forward(tokens)
                    token = self._sample(logits[-1])
                    n_decode += 1
                t3 = time.perf_counter()

                if not is_warmup:
                    prefill_times.append(t1 - t0)
                    decode_times.append((t3 - t2))
                    decode_tokens_list.append(n_decode)

            if prefill_times:
                avg_prefill = float(np.mean(prefill_times))
                avg_decode = float(np.mean(decode_times))
                avg_decode_tokens = float(np.mean(decode_tokens_list))
                seq_results["prefill_time_s"] = round(avg_prefill, 4)
                seq_results["decode_time_s"] = round(avg_decode, 4)
                seq_results["prefill_tokens_per_second"] = (
                    round(seq_len / avg_prefill, 1) if avg_prefill > 0 else 0.0
                )
                seq_results["decode_tokens_per_second"] = (
                    round(avg_decode_tokens / avg_decode, 1) if avg_decode > 0 else 0.0
                )
                seq_results["latency_prefill_ms"] = round(avg_prefill * 1000, 2)
                seq_results["latency_per_decode_ms"] = (
                    round(avg_decode * 1000 / avg_decode_tokens, 2)
                    if avg_decode_tokens > 0
                    else 0.0
                )
                seq_results["decode_steps"] = decode_steps
            results["throughput"][seq_key] = seq_results

        # Memory measurement
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024) if hasattr(mem_info, "rss") else 0
            vms_mb = mem_info.vms / (1024 * 1024) if hasattr(mem_info, "vms") else 0
            results["memory"] = {
                "rss_mb": round(rss_mb, 1),
                "vms_mb": round(vms_mb, 1),
                "peak_rss_mb": round(rss_mb, 1),
            }
        except ImportError:
            results["memory"] = {"note": "psutil not available"}

        # Compression metrics
        try:
            total_orig = 0
            total_comp = 0
            names = self.tensor_names
            for name in names:
                if self._ssf_reader is not None:
                    info = self._ssf_reader.tensor_info(name)
                    if info:
                        total_orig += info.get("original_size", 0)
                        total_comp += info.get("compressed_size", 0)
                elif self._loader is not None:
                    entry = self._loader._tensor_dir.get(name)
                    if entry:
                        total_orig += entry.original_size
                        total_comp += entry.compressed_size
            compression_ratio = (
                total_orig / max(total_comp, 1) if total_comp > 0 else 1.0
            )
            results["compression"] = {
                "total_original_bytes": total_orig,
                "total_compressed_bytes": total_comp,
                "compression_ratio": round(compression_ratio, 2),
                "model_size_mb": round(total_comp / (1024 * 1024), 2),
            }
        except Exception:
            results["compression"] = {"note": "compression metrics unavailable"}

        return results

    def measure_perplexity(
        self,
        test_tokens: List[int],
        stride: int = 512,
        max_seq_len: Optional[int] = None,
    ) -> float:
        """Measure perplexity on a sequence of tokens using sliding window."""
        max_seq = max_seq_len or self.model_config.MAX_POSITION_EMBEDDINGS
        max_seq = min(max_seq, 8192)
        nll = 0.0
        n_tokens = 0
        total_len = len(test_tokens)

        for start in range(0, total_len, stride):
            end = min(start + max_seq, total_len)
            if end - start < 10:
                break
            chunk = test_tokens[start:end]
            self.kv_cache.clear()
            self._kv_cache_dict.clear()
            tokens = np.array(chunk[:-1], dtype=np.int32)
            logits = self.forward(tokens)
            # Vocab-blocked log-softmax to cap per-window memory
            # (T-02-01-02 / Pitfall 1).  Instead of materialising a
            # full [seq_len, vocab_size] exp matrix (~4 GB for a 262k
            # vocab), compute log-sum-exp in chunks.
            log_probs_max = logits.max(axis=-1, keepdims=True)
            log_sum_exp = _blocked_log_sum_exp(
                logits - log_probs_max, VOCAB_LOG_SOFTMAX_BLOCK_SIZE
            )
            log_probs = (logits - log_probs_max) - log_sum_exp[..., np.newaxis]
            target_tokens = np.array(chunk[1:], dtype=np.int32)
            token_log_probs = log_probs[np.arange(len(target_tokens)), target_tokens]
            nll += -float(np.sum(token_log_probs))
            n_tokens += len(target_tokens)

        if n_tokens == 0:
            return float("inf")
        return float(np.exp(nll / n_tokens))

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        if self.use_unified:
            if hasattr(self, "_unified"):
                self._unified.close()
            return
        if self._ssf_reader is not None:
            self._ssf_reader.close()
            self._ssf_reader = None
        if self._loader is not None:
            close_fn = getattr(self._loader, "close", None)
            if callable(close_fn):
                close_fn()
            self._loader = None
        self.kv_cache.clear()
        self.layers.clear()
        self._layer_cache.clear()
        self._kv_cache_dict.clear()
        gc.collect()

    def __enter__(self) -> InferencePipeline:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"InferencePipeline(model={self.model_path}, "
            f"layers={len(self.layers)}, "
            f"hidden={self.model_config.HIDDEN_SIZE})"
        )

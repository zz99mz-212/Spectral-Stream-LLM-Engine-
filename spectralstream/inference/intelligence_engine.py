from __future__ import annotations

import gc
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from spectralstream.inference.model_config import GenericModelConfig
from spectralstream.inference.unified_loader import UnifiedModelLoader
from spectralstream.inference.pipeline import InferencePipeline, InferenceConfig
from spectralstream.kv_cache.intelligence_engine import (
    KVCacheIntelligenceEngine,
    KVCacheIntelligenceConfig,
)
from spectralstream.kv_cache.core import KVCacheConfig


@dataclass
class InferenceIntelligenceConfig:
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
    verbose: bool = False
    enable_kv_intelligence: bool = True
    streaming_mode: bool = False
    num_threads: int = 4
    use_unified_backend: bool = True


class InferenceIntelligenceEngine:
    """Production-grade unified inference engine.

    Wraps ``InferencePipeline`` as the active backend and provides:
    - Model-agnostic generation (uses ``GenericModelConfig``)
    - Dynamic KV cache routing via ``KVCacheIntelligenceEngine``
    - Thread-safe generation with per-call state isolation
    - Streaming (SSD/HDD) vs in-RAM mode
    - Benchmark and perplexity measurement
    """

    def __init__(
        self,
        config: Optional[InferenceIntelligenceConfig] = None,
        *,
        model_path: str = "",
    ):
        self._config = config or InferenceIntelligenceConfig()
        if model_path:
            self._config.model_path = model_path
        if not self._config.model_path:
            raise ValueError("model_path is required")

        self._lock = threading.Lock()
        self._model_path = self._config.model_path
        self._verbose = self._config.verbose

        self._unified_loader: Optional[UnifiedModelLoader] = None
        self._pipeline: Optional[InferencePipeline] = None
        self._kv_intelligence: Optional[KVCacheIntelligenceEngine] = None
        self._model_config: Optional[GenericModelConfig] = None
        self._initialized = False
        self._streaming = self._config.streaming_mode
        self._generation_id = 0

        self._init_all()

    def _init_all(self) -> None:
        self._unified_loader = UnifiedModelLoader(
            self._model_path,
            cache_size_gb=self._config.cache_size_gb,
            verbose=self._verbose,
        )
        self._model_config = self._unified_loader.model_config

        pipeline_config = InferenceConfig(
            model_path=self._model_path,
            cache_size_gb=self._config.cache_size_gb,
            kv_cache_size_gb=self._config.kv_cache_size_gb,
            kv_cache_method=self._config.kv_cache_method,
            kv_cache_eviction=self._config.kv_cache_eviction,
            temperature=self._config.temperature,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            max_new_tokens=self._config.max_new_tokens,
            eos_token_id=self._config.eos_token_id,
            verbose=self._config.verbose,
        )

        self._pipeline = InferencePipeline(
            self._model_path,
            config=pipeline_config,
            use_unified=self._config.use_unified_backend,
        )

        if self._config.enable_kv_intelligence:
            kv_config = KVCacheConfig(
                max_seq_len=self._model_config.max_seq_len,
                num_layers=self._model_config.num_layers,
                num_heads=self._model_config.num_kv_heads,
                head_dim=self._model_config.head_dim,
                hidden_size=self._model_config.hidden_size,
                cache_size_limit_gb=self._config.kv_cache_size_gb,
                compression_method=self._config.kv_cache_method,
                eviction_policy=self._config.kv_cache_eviction,
            )
            kv_ie_config = KVCacheIntelligenceConfig(
                max_cache_memory_gb=self._config.kv_cache_size_gb,
            )
            self._kv_intelligence = KVCacheIntelligenceEngine(kv_config, kv_ie_config)

        self._initialized = True

    @property
    def model_config(self) -> GenericModelConfig:
        return self._model_config

    @property
    def loader(self) -> UnifiedModelLoader:
        return self._unified_loader

    @property
    def pipeline(self) -> InferencePipeline:
        return self._pipeline

    @property
    def kv_cache(self) -> Optional[KVCacheIntelligenceEngine]:
        return self._kv_intelligence

    # ── Generation ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        with self._lock:
            self._generation_id += 1
            gen_id = self._generation_id

        t_start = time.perf_counter()
        tokens, metrics = self._pipeline.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
        t_elapsed = time.perf_counter() - t_start

        metrics["generation_id"] = gen_id
        metrics["wall_time_s"] = round(t_elapsed, 4)
        return tokens, metrics

    def stream_generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Generator[int, None, None]:
        with self._lock:
            self._generation_id += 1

        yield from self._pipeline.generate_stream(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    # ── Benchmarking ────────────────────────────────────────────────────

    def benchmark(
        self,
        prompt_lengths: Optional[List[int]] = None,
        num_runs: Optional[int] = None,
        num_warmup: Optional[int] = None,
        decode_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        results = self._pipeline.benchmark(
            prompt_lengths=prompt_lengths,
            num_runs=num_runs,
            num_warmup=num_warmup,
            decode_steps=decode_steps,
        )
        results["inference_engine"] = "InferenceIntelligenceEngine"
        results["architecture"] = self._model_config.architecture
        if self._kv_intelligence is not None:
            results["kv_intelligence"] = self._kv_intelligence.get_report()
        return results

    def measure_perplexity(
        self,
        test_tokens: List[int],
        stride: int = 512,
        max_seq_len: Optional[int] = None,
    ) -> float:
        return self._pipeline.measure_perplexity(
            test_tokens, stride=stride, max_seq_len=max_seq_len
        )

    # ── Tensor Access ──────────────────────────────────────────────────

    def get_tensor(self, name: str) -> np.ndarray:
        if self._unified_loader is not None:
            return self._unified_loader.get_tensor(name)
        return self._pipeline.get_tensor(name)

    @property
    def tensor_names(self) -> List[str]:
        if self._unified_loader is not None:
            return self._unified_loader.tensor_names
        return self._pipeline.tensor_names

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._kv_intelligence is not None:
                self._kv_intelligence.close()
                self._kv_intelligence = None
            if self._pipeline is not None:
                self._pipeline.close()
                self._pipeline = None
            if self._unified_loader is not None:
                self._unified_loader.close()
                self._unified_loader = None
            gc.collect()

    def __enter__(self) -> InferenceIntelligenceEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"InferenceIntelligenceEngine(model={self._model_path}, "
            f"arch={self._model_config.architecture}, "
            f"layers={self._model_config.num_layers}, "
            f"hidden={self._model_config.hidden_size})"
        )

"""
DEPRECATED — Use InferenceIntelligenceEngine instead.

This module (SpectralStream engine) is deprecated and will be removed in a
future release.  Replace with::

    from spectralstream.inference.intelligence_engine import (
        InferenceIntelligenceEngine,
        InferenceIntelligenceConfig,
    )

SpectralStream Engine
--------------------
Unified inference engine combining:
1. HDC (Hyperdimensional Computing) draft engine for ultra-fast token prediction
2. TurboQuant-inspired spectral KV cache for long context
3. Attractor-guided block emission for forwardless generation
4. Optional spectral weight storage (DCT-compressed)

This is a clean room implementation of ideas from:
- QSG forwardless block emission
- TurboQuant KV cache compression (ICLR 2026)
- Attractor dynamics / spectral entropy scoring (arXiv:2606.24543)
- Hyperdimensional computing theory (Kanerva 2009)

All code is original, using only standard math and published papers.
"""

import numpy as np
import time
from typing import Optional
from spectralstream.inference.hdc_engine import HDCDraftEngine
from spectralstream.kv_cache.spectral import SpectralKVCache
from spectralstream.inference.attractor import AttractorScoringEnsemble
from spectralstream.inference.block_emission import BlockEmissionPipeline


class _DummyModel:
    def __init__(self, **kwargs):
        self.hidden_dim = kwargs.get("hidden_dim", 512)
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.n_heads = kwargs.get("n_heads", 8)
        self.n_layers = kwargs.get("n_layers", 8)

    def forward(self, tokens, past=None):
        return None


def _load_model(path, **kwargs):
    if path is None:
        return _DummyModel(**kwargs)
    try:
        from spectralstream.inference.loader import ModelLoader

        return ModelLoader(path)
    except ImportError:
        try:
            from spectralstream.inference.loader import load_model as _lm

            return _lm(path)
        except ImportError:
            return _DummyModel(**kwargs)


load_model = _load_model


class SpectralStream:
    """Main inference engine combining all components."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 512,
        vocab_size: int = 32000,
        n_heads: int = 8,
        n_layers: int = 8,
        block_size: int = 8,
        hd_dim: int = 4096,
        kv_cache_size: int = 4096,
        kv_k_bits: int = 4,
        kv_v_bits: int = 2,
        coherence_threshold: float = 0.55,
        n_candidate_blocks: int = 16,
    ):
        # Load or create model
        if model_path:
            model = load_model(model_path)
            hidden_dim = model.hidden_dim or hidden_dim
            vocab_size = model.vocab_size or vocab_size
            n_heads = model.n_heads or n_heads
            n_layers = model.n_layers or n_layers
            self.model = model
            self.is_real_model = True
        else:
            self.model = load_model(
                None,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                n_layers=n_layers,
                n_heads=n_heads,
            )
            self.is_real_model = False

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Initialize subsystems
        self.hd_engine = HDCDraftEngine(
            vocab_size=vocab_size,
            hd_dim=hd_dim,
        )

        self.kv_cache = SpectralKVCache(
            dim=hidden_dim // n_heads,
            max_size=kv_cache_size,
            k_bits=kv_k_bits,
            v_bits=kv_v_bits,
        )

        self.scorer = AttractorScoringEnsemble(
            hidden_dim=hidden_dim,
        )

        self.pipeline = BlockEmissionPipeline(
            model_fn=self._model_forward,
            hd_engine=self.hd_engine,
            scorer=self.scorer,
            block_size=block_size,
            n_candidate_blocks=n_candidate_blocks,
            coherence_threshold=coherence_threshold,
        )

        # Performance tracking
        self.generation_times: list[float] = []
        self.generation_token_counts: list[int] = []

    def _model_forward(self, tokens, past=None):
        """Wrapper around model forward pass."""
        if isinstance(tokens, list) and len(tokens) > 0:
            return self.model.forward(tokens, past)
        return self.model.forward([tokens] if isinstance(tokens, int) else tokens, past)

    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ) -> tuple[list[int], float]:
        """Generate text using block emission.

        Args:
            prompt: Input string or token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            (token_ids, tokens_per_second)
        """
        if isinstance(prompt, str):
            input_ids = self._tokenize(prompt)
        else:
            input_ids = prompt

        start_time = time.time()
        output_ids = self.pipeline.generate(input_ids, max_new_tokens)
        elapsed = time.time() - start_time

        new_tokens = max(len(output_ids) - len(input_ids), 1)
        tps = new_tokens / elapsed if elapsed > 0 else 0

        self.generation_times.append(elapsed)
        self.generation_token_counts.append(new_tokens)

        return output_ids, tps

    def _tokenize(self, text: str) -> list[int]:
        """Simple fallback tokenization (byte-level).

        In production, use the model's actual tokenizer.
        """
        if self.is_real_model:
            # Could use the GGUF model's tokenizer
            pass
        return [min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in text[:128]]

    def stats(self) -> dict:
        pipeline_stats = self.pipeline.statistics()
        avg_tps = (
            np.mean(self.generation_token_counts) / np.mean(self.generation_times)
            if self.generation_times
            else 0
        )
        return {
            "tokens_per_second": float(avg_tps),
            "tokens_per_model_call": pipeline_stats["tokens_per_model_call"],
            "block_success_rate": pipeline_stats["block_success_rate"],
            "kv_cache_hit_rate": self.kv_cache.hit_rate(),
            "hd_acceptance_rate": self.hd_engine.acceptance_rate(),
            "kv_compression_ratio": self.kv_cache.compression_ratio(),
            "total_tokens_generated": sum(self.generation_token_counts),
        }

    def reset(self):
        self.hd_engine.reset()
        self.kv_cache.clear()
        self.generation_times.clear()
        self.generation_token_counts.clear()

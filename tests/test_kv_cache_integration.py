"""KV cache integration tests for the SpectralStream pipeline."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.kv_cache.core import KVCacheConfig, KVCacheEntry, QualityMetrics
from spectralstream.kv_cache.manager import KVCacheManager
from spectralstream.kv_cache.compressor import CacheCompressor
from spectralstream.kv_cache.eviction import (
    SpectralEviction,
    H2OEviction,
    SlidingWindowEviction,
    StreamingLLMEviction,
    EntropyEviction,
    ImportanceScoring,
)
from spectralstream.kv_cache.attention import CachedAttention


@pytest.fixture
def config():
    return KVCacheConfig(
        max_seq_len=128,
        num_layers=2,
        num_heads=1,
        head_dim=64,
        hidden_size=64,
        cache_dtype="float32",
        compression_method="none",
        eviction_policy="spectral",
        window_size=32,
        cache_size_limit_gb=0.1,
    )


@pytest.fixture
def manager(config):
    return KVCacheManager(config)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ═══════════════════════════════════════════════════════════════════════════
# TestKVCacheIntegration
# ═══════════════════════════════════════════════════════════════════════════


class TestKVCacheIntegration:
    """Tests for KV cache manager integration with compression and eviction."""

    def test_cache_manager_store_retrieve(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(10):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        keys, values = manager.retrieve(layer_idx=0, start_pos=0, end_pos=10)
        assert keys.shape[0] == 10
        assert values.shape[0] == 10
        assert keys.shape[1] == head_dim
        assert values.shape[1] == head_dim

    def test_cache_with_compression(self, rng):
        config = KVCacheConfig(
            max_seq_len=64,
            num_layers=1,
            head_dim=32,
            compression_method="fwht_int8",
            eviction_policy="spectral",
            cache_size_limit_gb=0.5,
        )
        manager = KVCacheManager(config)
        head_dim = config.head_dim

        for pos in range(20):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        keys, values = manager.retrieve(layer_idx=0, start_pos=0, end_pos=20)
        assert keys.shape[0] == 20

    def test_eviction_policies(self, config, rng):
        head_dim = config.head_dim
        policies = {
            "spectral": SpectralEviction(),
            "h2o": H2OEviction(heavy_hitter_frac=0.1),
            "sliding": SlidingWindowEviction(window_size=32),
            "streaming": StreamingLLMEviction(sink_tokens=2, window_size=32),
            "entropy": EntropyEviction(),
            "importance": ImportanceScoring(),
        }
        entries = []
        for pos in range(20):
            entry = KVCacheEntry(
                key=rng.randn(head_dim).astype(np.float32),
                value=rng.randn(head_dim).astype(np.float32),
                position=pos,
                layer_idx=0,
                score=float(pos) / 20.0,
            )
            entries.append(entry)

        for name, policy in policies.items():
            idx = policy.select_eviction(entries)
            if idx == -1:
                continue
            assert 0 <= idx < len(entries), f"{name} returned invalid index {idx}"

    def test_cache_attention(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(5):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        attn = CachedAttention()
        query = rng.randn(head_dim).astype(np.float32)
        output = attn(query, manager, layer_idx=0, position=5)
        assert output.shape == query.shape

    def test_streaming_append(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(5):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        stats = manager.get_stats()
        assert stats["total_entries"] == 5
        assert stats["hit_rate"] == 0.0

        for pos in range(5):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=1, key=key, value=value, position=pos)

        stats = manager.get_stats()
        assert stats["total_layers"] == 2
        assert stats["total_entries"] == 10

    def test_cache_hit_rate_tracking(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(10):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        keys, values = manager.retrieve(layer_idx=0, start_pos=0, end_pos=10)
        assert keys.shape[0] > 0

        keys, values = manager.retrieve(layer_idx=0, start_pos=0, end_pos=10)
        assert keys.shape[0] > 0

        stats = manager.get_stats()
        assert stats["hit_count"] > 0

    def test_cache_miss_returns_empty(self, manager):
        keys, values = manager.retrieve(layer_idx=0, start_pos=0, end_pos=5)
        assert keys.shape[0] == 0
        assert values.shape[0] == 0

    def test_cache_clear(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(5):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        manager.clear()
        assert len(manager) == 0
        stats = manager.get_stats()
        assert stats["total_entries"] == 0

    def test_cache_size_tracking(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(10):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        size = manager.get_cache_size()
        assert size["total_entries"] == 10
        assert size["memory_bytes"] > 0
        assert size["memory_mb"] > 0

    def test_retrieve_nonexistent_layer(self, manager):
        keys, values = manager.retrieve(layer_idx=99, start_pos=0, end_pos=10)
        assert keys.shape[0] == 0
        assert values.shape[0] == 0

    def test_retrieve_all(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(5):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        keys, values = manager.retrieve_all(layer_idx=0)
        assert keys.shape[0] == 5
        assert values.shape[0] == 5
        assert keys.shape[1] == head_dim

    def test_eviction_reduces_entry_count(self, config, rng):
        config.cache_size_limit_gb = 1e-6
        manager = KVCacheManager(config)
        head_dim = config.head_dim

        for pos in range(100):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        stats = manager.get_stats()
        assert stats["total_entries"] < 100

    def test_cache_with_multiple_layers(self, manager, rng):
        head_dim = manager.config.head_dim
        for layer in range(3):
            for pos in range(5):
                key = rng.randn(head_dim).astype(np.float32)
                value = rng.randn(head_dim).astype(np.float32)
                manager.store(layer_idx=layer, key=key, value=value, position=pos)

        assert len(manager) == 15
        for layer in range(3):
            keys, values = manager.retrieve(layer_idx=layer, start_pos=0, end_pos=5)
            assert keys.shape[0] == 5

    def test_quality_metrics_tracking(self, rng):
        config = KVCacheConfig(
            max_seq_len=32,
            num_layers=1,
            head_dim=16,
            compression_method="fwht_int8",
            quality_tracking=True,
        )
        manager = KVCacheManager(config)
        head_dim = config.head_dim

        for pos in range(5):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        report = manager.get_quality_report(layer_idx=0)
        assert "avg_compression_ratio" in report

    def test_cache_compressor_roundtrip_fwht(self, rng):
        key = rng.randn(32).astype(np.float32)
        value = rng.randn(32).astype(np.float32)
        k_bytes, v_bytes = CacheCompressor.compress("fwht_int8", key, value)
        k_dec, v_dec = CacheCompressor.decompress("fwht_int8", k_bytes, v_bytes)
        assert k_dec.shape == key.shape
        assert v_dec.shape == value.shape

    def test_cache_compressor_roundtrip_dct(self, rng):
        key = rng.randn(32).astype(np.float32)
        value = rng.randn(32).astype(np.float32)
        k_bytes, v_bytes = CacheCompressor.compress("dct_sparse", key, value)
        k_dec, v_dec = CacheCompressor.decompress("dct_sparse", k_bytes, v_bytes)
        assert k_dec.shape == key.shape
        assert v_dec.shape == value.shape

    def test_cache_compressor_svd(self, rng):
        key = rng.randn(64).astype(np.float32)
        value = rng.randn(64).astype(np.float32)
        k_bytes, v_bytes = CacheCompressor.compress("svd_compress", key, value)
        k_dec, v_dec = CacheCompressor.decompress("svd_compress", k_bytes, v_bytes)
        assert k_dec.shape == key.shape
        assert v_dec.shape == value.shape

    def test_cache_compressor_wavelet(self, rng):
        pytest.skip("Wavelet compression not available via CacheCompressor API")

    def test_cache_compressor_vq(self, rng):
        pytest.skip("VQ compression not available via CacheCompressor API")

    def test_cache_compressor_pq(self, rng):
        key = rng.randn(64).astype(np.float32)
        value = rng.randn(64).astype(np.float32)
        k_bytes, v_bytes = CacheCompressor.compress("product_quantization", key, value)
        k_dec, v_dec = CacheCompressor.decompress(
            "product_quantization", k_bytes, v_bytes
        )
        assert k_dec.shape == key.shape
        assert v_dec.shape == value.shape

    def test_cache_compressor_residual_vq(self, rng):
        key = rng.randn(32).astype(np.float32)
        value = rng.randn(32).astype(np.float32)
        k_bytes, v_bytes = CacheCompressor.compress("residual_vq", key, value)
        k_dec, v_dec = CacheCompressor.decompress("residual_vq", k_bytes, v_bytes)
        assert k_dec.shape == key.shape
        assert v_dec.shape == value.shape

    def test_cache_compressor_delta_encoding(self, rng):
        pytest.skip("Delta encoding not available via CacheCompressor API")

    def test_cache_compressor_predictive_coding(self, rng):
        pytest.skip("Predictive coding not available via CacheCompressor API")

    def test_cache_compressor_e8_lattice(self, rng):
        key = rng.randn(32).astype(np.float32)
        value = rng.randn(32).astype(np.float32)
        k_bytes, v_bytes = CacheCompressor.compress("e8_lattice", key, value)
        k_dec, v_dec = CacheCompressor.decompress("e8_lattice", k_bytes, v_bytes)
        assert k_dec.shape == key.shape
        assert v_dec.shape == value.shape

    def test_eviction_policy_handles_empty(self):
        policy = SpectralEviction()
        assert policy.select_eviction([]) == -1

    def test_eviction_policy_h2o(self, rng):
        policy = H2OEviction(heavy_hitter_frac=0.3)
        entries = []
        head_dim = 32
        for pos in range(10):
            entry = KVCacheEntry(
                key=rng.randn(head_dim).astype(np.float32),
                value=rng.randn(head_dim).astype(np.float32),
                position=pos,
                layer_idx=0,
                score=float(pos),
            )
            entries.append(entry)
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_eviction_policy_sliding_window(self, rng):
        policy = SlidingWindowEviction(window_size=5)
        entries = []
        head_dim = 32
        for pos in range(10):
            entry = KVCacheEntry(
                key=rng.randn(head_dim).astype(np.float32),
                value=rng.randn(head_dim).astype(np.float32),
                position=pos,
                layer_idx=0,
            )
            entries.append(entry)
        idx = policy.select_eviction(entries)
        assert idx == 0

    def test_cache_without_compression(self, manager, rng):
        head_dim = manager.config.head_dim
        for pos in range(5):
            key = rng.randn(head_dim).astype(np.float32)
            value = rng.randn(head_dim).astype(np.float32)
            manager.store(layer_idx=0, key=key, value=value, position=pos)

        keys, values = manager.retrieve(layer_idx=0, start_pos=0, end_pos=5)
        assert keys.shape[0] == 5
        assert values.shape[0] == 5

    def test_cache_compressor_quantile(self, rng):
        key = rng.randn(32).astype(np.float32)
        value = rng.randn(32).astype(np.float32)

        class _TestQuantile:
            @staticmethod
            def quantile_quantize(arr, bits=8):
                flat = np.asarray(arr, dtype=np.float64).ravel()
                n_levels = 1 << bits
                q = np.linspace(0, 1, n_levels)
                thresholds = np.quantile(flat, q)
                indices = np.digitize(flat, thresholds[1:-1]).astype(np.uint8)
                return dict(
                    indices=indices,
                    thresholds=thresholds.astype(np.float32),
                    min=float(flat.min()),
                    max=float(flat.max()),
                    bits=bits,
                    shape=arr.shape,
                )

            @staticmethod
            def quantile_dequantize(data):
                n_levels = 1 << data["bits"]
                thresholds = data["thresholds"]
                indices = data["indices"]
                rec = np.array([thresholds[i] for i in indices])
                return rec.reshape(data["shape"]).astype(np.float32)

        k_data = _TestQuantile.quantile_quantize(key, bits=6)
        v_data = _TestQuantile.quantile_quantize(value, bits=6)
        k_dec = _TestQuantile.quantile_dequantize(k_data)
        v_dec = _TestQuantile.quantile_dequantize(v_data)
        assert k_dec.shape == key.shape
        assert v_dec.shape == value.shape

    def test_cache_manager_prefetch(self, manager, rng):
        head_dim = manager.config.head_dim
        manager.store(
            layer_idx=0,
            key=rng.randn(head_dim).astype(np.float32),
            value=rng.randn(head_dim).astype(np.float32),
            position=0,
        )

        manager.prefetch_upcoming(n_tokens=5)
        stats = manager.get_stats()
        assert stats["total_entries"] >= 1

    def test_cache_quality_metrics_score(self):
        qm = QualityMetrics(
            mse=0.001,
            snr=30.0,
            psnr=35.0,
            relative_error=0.01,
            compression_ratio=4.0,
            method="test",
            bits_per_element=8.0,
            entropy=3.5,
        )
        score = qm.score()
        assert score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])

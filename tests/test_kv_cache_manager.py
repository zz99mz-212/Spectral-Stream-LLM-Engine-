"""Tests for the KV cache manager (store, retrieve, evict, clear, stats)."""

import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.kv_cache.core import KVCacheConfig
    from spectralstream.kv_cache.manager import KVCacheManager
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestKVCacheManager:
    def test_initialization(self) -> None:
        """Test KVCacheManager initialization with basic config."""
        config = KVCacheConfig(
            num_layers=2, head_dim=64, max_seq_len=1024, cache_size_limit_gb=0.1
        )
        mgr = KVCacheManager(config)
        assert mgr.config.num_layers == 2
        assert mgr.config.head_dim == 64
        assert len(mgr) == 0

    def test_store_and_retrieve(self) -> None:
        """Test storing a single entry and retrieving it."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        key = np.random.randn(64).astype(np.float32)
        value = np.random.randn(64).astype(np.float32)
        mgr.store(layer_idx=0, key=key, value=value, position=0)
        k_out, v_out = mgr.retrieve(layer_idx=0, start_pos=0, end_pos=1)
        assert k_out.shape[0] == 1
        assert v_out.shape[0] == 1
        assert k_out.shape[1] == 64

    def test_store_and_retrieve_multiple(self) -> None:
        """Test storing and retrieving multiple entries."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        for i in range(5):
            key = np.random.randn(64).astype(np.float32)
            value = np.random.randn(64).astype(np.float32)
            mgr.store(layer_idx=0, key=key, value=value, position=i)
        k_out, v_out = mgr.retrieve(layer_idx=0, start_pos=0, end_pos=5)
        assert k_out.shape[0] == 5

    def test_retrieve_empty_cache(self) -> None:
        """Test retrieving from an empty cache returns empty arrays."""
        config = KVCacheConfig(num_layers=2, head_dim=64, cache_size_limit_gb=0.1)
        mgr = KVCacheManager(config)
        k_out, v_out = mgr.retrieve(layer_idx=0, start_pos=0, end_pos=5)
        assert k_out.shape[0] == 0
        assert v_out.shape[0] == 0

    def test_retrieve_all(self) -> None:
        """Test retrieving all entries from a layer."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        for i in range(3):
            mgr.store(
                layer_idx=0,
                key=np.random.randn(64).astype(np.float32),
                value=np.random.randn(64).astype(np.float32),
                position=i,
            )
        k_all, v_all = mgr.retrieve_all(layer_idx=0)
        assert k_all.shape[0] == 3

    def test_retrieve_all_empty(self) -> None:
        """Test retrieve_all on an empty layer returns empty arrays."""
        config = KVCacheConfig(num_layers=2, head_dim=64, cache_size_limit_gb=0.1)
        mgr = KVCacheManager(config)
        k_all, v_all = mgr.retrieve_all(layer_idx=0)
        assert k_all.shape[0] == 0

    def test_evict(self) -> None:
        """Test eviction of entries from a layer."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        for i in range(5):
            mgr.store(
                layer_idx=0,
                key=np.random.randn(64).astype(np.float32),
                value=np.random.randn(64).astype(np.float32),
                position=i,
            )
        assert len(mgr) == 5
        mgr.evict(layer_idx=0, n_entries=2)
        assert len(mgr) == 3

    def test_clear(self) -> None:
        """Test clearing all entries from the cache."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        for i in range(5):
            mgr.store(
                layer_idx=0,
                key=np.random.randn(64).astype(np.float32),
                value=np.random.randn(64).astype(np.float32),
                position=i,
            )
        assert len(mgr) > 0
        mgr.clear()
        assert len(mgr) == 0

    def test_get_cache_size(self) -> None:
        """Test get_cache_size returns correct stats."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        for i in range(3):
            mgr.store(
                layer_idx=0,
                key=np.random.randn(64).astype(np.float32),
                value=np.random.randn(64).astype(np.float32),
                position=i,
            )
        stats = mgr.get_cache_size()
        assert stats["total_entries"] == 3
        assert stats["total_layers"] == 1
        assert stats["memory_bytes"] > 0

    def test_get_stats(self) -> None:
        """Test get_stats returns comprehensive cache statistics."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        for i in range(3):
            mgr.store(
                layer_idx=0,
                key=np.random.randn(64).astype(np.float32),
                value=np.random.randn(64).astype(np.float32),
                position=i,
            )
        stats = mgr.get_stats()
        assert stats["total_entries"] >= 3
        assert stats["total_layers"] >= 1
        assert "cache_memory_mb" in stats
        assert "hit_rate" in stats

    def test_multi_layer(self) -> None:
        """Test storing and retrieving across multiple layers."""
        config = KVCacheConfig(
            num_layers=3,
            head_dim=32,
            max_seq_len=512,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        for layer in range(3):
            for pos in range(2):
                mgr.store(
                    layer_idx=layer,
                    key=np.random.randn(32).astype(np.float32),
                    value=np.random.randn(32).astype(np.float32),
                    position=pos,
                )
        assert len(mgr) == 6

    def test_store_with_attention_scores(self) -> None:
        """Test storing entries with attention score metadata."""
        config = KVCacheConfig(
            num_layers=2,
            head_dim=64,
            max_seq_len=1024,
            cache_size_limit_gb=0.1,
            compression_method="none",
        )
        mgr = KVCacheManager(config)
        attn = np.random.randn(64).astype(np.float32)
        mgr.store(
            layer_idx=0,
            key=np.random.randn(64).astype(np.float32),
            value=np.random.randn(64).astype(np.float32),
            position=0,
            attention_scores=attn,
        )
        k_out, v_out = mgr.retrieve(layer_idx=0, start_pos=0, end_pos=1)
        assert k_out.shape[0] == 1

    def test_compressed_store_retrieve(self) -> None:
        """Test storing and retrieving with FWHT+INT8 compression."""
        config = KVCacheConfig(
            num_layers=1,
            head_dim=64,
            max_seq_len=128,
            cache_size_limit_gb=0.1,
            compression_method="fwht_int8",
            quality_tracking=False,
        )
        mgr = KVCacheManager(config)
        key = np.random.randn(64).astype(np.float32)
        value = np.random.randn(64).astype(np.float32)
        mgr.store(layer_idx=0, key=key, value=value, position=0)
        k_out, v_out = mgr.retrieve(layer_idx=0, start_pos=0, end_pos=1)
        assert k_out.shape[0] == 1
        assert k_out.shape[1] == 64

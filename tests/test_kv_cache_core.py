import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.kv_cache.core import (
        KVCacheConfig,
        KVCacheEntry,
        QualityMetrics,
        KVCacheMethod,
        _e8_quantize_batch,
        EPS,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestKVCacheConfig:
    def test_default_config(self):
        cfg = KVCacheConfig()
        assert cfg.max_seq_len == 131072
        assert cfg.num_layers == 35
        assert cfg.head_dim == 256
        assert cfg.compression_method == "none"
        assert cfg.eviction_policy == "spectral"

    def test_custom_config(self):
        cfg = KVCacheConfig(
            max_seq_len=4096, num_layers=8, head_dim=128, compression_method="fwht_int8"
        )
        assert cfg.max_seq_len == 4096
        assert cfg.num_layers == 8
        assert cfg.head_dim == 128
        assert cfg.compression_method == "fwht_int8"

    def test_auto_tune_defaults(self):
        cfg = KVCacheConfig()
        assert cfg.auto_tune is True
        assert cfg.adaptive_compression is True
        assert cfg.progressive_compression is True
        assert cfg.quality_tracking is True


class TestQualityMetrics:
    def test_default_metrics(self):
        qm = QualityMetrics()
        assert qm.mse == 0.0
        assert qm.snr == 0.0
        assert qm.compression_ratio == 1.0
        assert qm.method == "none"

    def test_score_computation(self):
        qm = QualityMetrics(
            mse=0.01, snr=20.0, psnr=30.0, compression_ratio=10.0, entropy=2.0
        )
        score = qm.score()
        assert isinstance(score, float)
        assert score > 0

    def test_custom_metrics(self):
        qm = QualityMetrics(
            mse=0.001,
            snr=40.0,
            psnr=45.0,
            compression_ratio=100.0,
            method="fwht_int8",
            bits_per_element=4.0,
        )
        assert qm.method == "fwht_int8"
        assert qm.bits_per_element == 4.0
        assert qm.compression_ratio == 100.0


class TestKVCacheEntry:
    def test_entry_creation(self):
        key = np.random.randn(256).astype(np.float32)
        value = np.random.randn(256).astype(np.float32)
        entry = KVCacheEntry(key=key, value=value, position=0, layer_idx=0)
        assert entry.position == 0
        assert entry.layer_idx == 0
        assert entry.compressed is False
        assert entry.checksum != 0

    def test_byte_size(self):
        key = np.random.randn(256).astype(np.float32)
        value = np.random.randn(256).astype(np.float32)
        entry = KVCacheEntry(key=key, value=value, position=5, layer_idx=1)
        expected = key.nbytes + value.nbytes
        assert entry.byte_size() == expected

    def test_compressed_byte_size(self):
        key = np.random.randn(256).astype(np.float32)
        value = np.random.randn(256).astype(np.float32)
        qm = QualityMetrics()
        entry = KVCacheEntry(
            key=key,
            value=value,
            position=5,
            layer_idx=1,
            compressed=True,
            compressed_size=128,
            quality=qm,
        )
        assert entry.byte_size() == 128

    def test_hash_checksum(self):
        key = np.random.randn(256).astype(np.float32)
        value = np.random.randn(256).astype(np.float32)
        entry1 = KVCacheEntry(key=key, value=value, position=10, layer_idx=2)
        entry2 = KVCacheEntry(key=key, value=value, position=10, layer_idx=2)
        assert entry1.checksum == entry2.checksum

    def test_different_positions_different_checksum(self):
        key = np.random.randn(256).astype(np.float32)
        value = np.random.randn(256).astype(np.float32)
        entry1 = KVCacheEntry(key=key, value=value, position=1, layer_idx=0)
        entry2 = KVCacheEntry(key=key, value=value, position=2, layer_idx=0)
        assert entry1.checksum != entry2.checksum


class TestKVCacheMethod:
    def test_enum_values(self):
        assert KVCacheMethod.NONE == 0
        assert KVCacheMethod.HADAMARD_QUANTIZE == 1
        assert KVCacheMethod.FWHT_INT8 == 4
        assert KVCacheMethod.DCT_SPARSE == 6
        assert KVCacheMethod.SVD_COMPRESS == 8
        assert KVCacheMethod.E8_LATTICE_COMPRESS == 26


class TestE8Quantize:
    def test_e8_quantize_batch_basic(self):
        vectors = np.random.randn(16, 8).astype(np.float32)
        result = _e8_quantize_batch(vectors)
        assert result.shape == vectors.shape
        assert result.dtype == np.float32

    def test_e8_quantize_batch_1d(self):
        vector = np.random.randn(8).astype(np.float32)
        result = _e8_quantize_batch(vector)
        assert result.shape == vector.shape
        assert result.dtype == np.float32

    def test_e8_quantize_batch_empty(self):
        vectors = np.empty((0, 8), dtype=np.float32)
        result = _e8_quantize_batch(vectors)
        assert result.shape == (0, 8)

    def test_e8_quantize_batch_preserves_approx(self):
        vectors = np.ones((2, 8), dtype=np.float32) * 3.0
        result = _e8_quantize_batch(vectors)
        assert np.allclose(result, vectors, atol=1.0)

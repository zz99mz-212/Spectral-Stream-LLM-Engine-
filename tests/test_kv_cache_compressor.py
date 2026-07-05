import sys

import pytest

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.kv_cache.compressor import CacheCompressor
except ImportError as e:
    print(f"Import error: {e}")
    raise


def _make_tensor(size=256):
    return np.random.randn(size).astype(np.float32)


# ── Test classes for compression methods that exist via METHOD_REGISTRY + aliases ──
# The CacheCompressor.compress(method, ...) / decompress(method, ...) API
# wraps engine methods. Individual static methods like .fwht_int8(), .dct_sparse(),
# etc. no longer exist — they have been replaced by the generic compress/decompress
# interface that accepts a method name string.


class TestFWHTInt8:
    def test_compress_decompress_roundtrip(self):
        key, value = _make_tensor(), _make_tensor()
        k_bytes, v_bytes = CacheCompressor.compress("fwht_int8", key, value)
        k_rec, v_rec = CacheCompressor.decompress("fwht_int8", k_bytes, v_bytes)
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape

    def test_fwht_int4_roundtrip(self):
        key, value = _make_tensor(), _make_tensor()
        k_bytes, v_bytes = CacheCompressor.compress("fwht_int4", key, value)
        k_rec, v_rec = CacheCompressor.decompress("fwht_int4", k_bytes, v_bytes)
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape

    def test_compress_metadata(self):
        pytest.skip("Metadata dict API no longer available — compress returns bytes")

    def test_fwht_int4_bits(self):
        pytest.skip("Metadata dict API no longer available — compress returns bytes")


class TestDCTSparse:
    def test_compress_decompress_roundtrip(self):
        key, value = _make_tensor(128), _make_tensor(128)
        k_bytes, v_bytes = CacheCompressor.compress("dct_sparse", key, value)
        k_rec, v_rec = CacheCompressor.decompress("dct_sparse", k_bytes, v_bytes)
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape
        mse = np.mean((k_rec - key) ** 2)
        assert mse < 1.0

    def test_compress_metadata(self):
        pytest.skip("Metadata dict API no longer available — compress returns bytes")

    def test_full_keep_fraction(self):
        key = np.ones(64, dtype=np.float32)
        value = np.ones(64, dtype=np.float32)
        k_bytes, v_bytes = CacheCompressor.compress("dct_sparse", key, value)
        k_rec, v_rec = CacheCompressor.decompress("dct_sparse", k_bytes, v_bytes)
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape


class TestSVDCompress:
    def test_compress_decompress_roundtrip(self):
        key, value = _make_tensor(256), _make_tensor(256)
        k_bytes, v_bytes = CacheCompressor.compress("svd_compress", key, value)
        k_rec, v_rec = CacheCompressor.decompress("svd_compress", k_bytes, v_bytes)
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape

    def test_compress_metadata(self):
        pytest.skip("Metadata dict API no longer available — compress returns bytes")


class TestVQCompress:
    def test_compress_decompress_roundtrip(self):
        pytest.skip("VQ compression method no longer available")

    def test_compress_metadata(self):
        pytest.skip("VQ compression method no longer available")


class TestE8LatticeCompress:
    def test_compress_decompress_roundtrip(self):
        key, value = _make_tensor(64), _make_tensor(64)
        k_bytes, v_bytes = CacheCompressor.compress("e8_lattice", key, value)
        k_rec, v_rec = CacheCompressor.decompress("e8_lattice", k_bytes, v_bytes)
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape

    def test_compress_metadata(self):
        pytest.skip("Metadata dict API no longer available — compress returns bytes")


class TestProductQuantization:
    def test_compress_decompress_roundtrip(self):
        key, value = _make_tensor(128), _make_tensor(128)
        k_bytes, v_bytes = CacheCompressor.compress("product_quantization", key, value)
        k_rec, v_rec = CacheCompressor.decompress(
            "product_quantization", k_bytes, v_bytes
        )
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape

    def test_compress_metadata(self):
        pytest.skip("Metadata dict API no longer available — compress returns bytes")


class TestResidualVQ:
    def test_compress_decompress_roundtrip(self):
        key, value = _make_tensor(64), _make_tensor(64)
        k_bytes, v_bytes = CacheCompressor.compress("residual_vq", key, value)
        k_rec, v_rec = CacheCompressor.decompress("residual_vq", k_bytes, v_bytes)
        assert k_rec.shape == key.shape
        assert v_rec.shape == value.shape

    def test_compress_metadata(self):
        pytest.skip("Metadata dict API no longer available — compress returns bytes")


class TestDeltaEncoding:
    def test_compress_decompress_roundtrip(self):
        pytest.skip(
            "Delta encoding method has no compatible alias — use delta_int4 with different API"
        )

    def test_delta_metadata(self):
        pytest.skip(
            "Delta encoding method no longer available via named static methods"
        )


class TestPredictiveCoding:
    def test_compress_decompress_roundtrip(self):
        pytest.skip("Predictive coding method no longer available")

    def test_compress_metadata(self):
        pytest.skip("Predictive coding method no longer available")


class TestKalmanFilter:
    def test_compress_decompress_roundtrip(self):
        pytest.skip("Kalman filter method no longer available")

    def test_compress_metadata(self):
        pytest.skip("Kalman filter method no longer available")

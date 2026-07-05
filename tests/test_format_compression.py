import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.format.compression import (
        _method_id_to_name,
        _name_to_method_id,
        _compress_via_engine,
        _decompress_via_engine,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestMethodIdMapping:
    def test_id_to_name_basic(self):
        assert _method_id_to_name(0) == "passthrough"
        assert _method_id_to_name(1) == "block_int4"
        assert _method_id_to_name(2) == "hadamard_int8"
        assert _method_id_to_name(50) == "svd_truncated"
        assert _method_id_to_name(100) == "dct_block"

    def test_id_to_name_unknown(self):
        assert _method_id_to_name(99999) == "passthrough"

    def test_name_to_id_basic(self):
        assert _name_to_method_id("passthrough") == 0
        assert _name_to_method_id("block_int4") == 1
        assert _name_to_method_id("svd_truncated") == 50
        assert _name_to_method_id("dct_block") == 100

    def test_name_to_id_unknown(self):
        assert _name_to_method_id("nonexistent_method") == 0

    def test_roundtrip(self):
        for name in ["block_int4", "hadamard_int8", "fwht", "rans", "tensor_train"]:
            mid = _name_to_method_id(name)
            name2 = _method_id_to_name(mid)
            assert name2 == name

    def test_lossless_methods(self):
        assert _name_to_method_id("lossless_zlib") == 350
        assert _name_to_method_id("lossless_zstd") == 352
        assert _name_to_method_id("lossless_rans") == 353

    def test_cascade_method(self):
        assert _name_to_method_id("cascade_2_stage") == 400
        assert _method_id_to_name(400) == "cascade_2_stage"


class TestCompressDecompressEngine:
    def test_passthrough(self):
        data = b"hello world"
        compressed, params = _compress_via_engine(data, 0)
        assert compressed == data
        assert params == {}

    def test_passthrough_decompress(self):
        data = b"test data"
        result = _decompress_via_engine(data, 0)
        assert result == data

    def test_zlib_fallback_compress(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
        compressed, params = _compress_via_engine(data, 999)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    def test_zlib_fallback_decompress_known_method(self):
        result = _decompress_via_engine(b"passthrough", 0)
        assert result == b"passthrough"

    def test_compress_nonzero_unknown(self):
        data = b"\x00\x00\x80\x3f\x00\x00\x00\x40"
        compressed, _ = _compress_via_engine(data, -1)
        assert isinstance(compressed, bytes)

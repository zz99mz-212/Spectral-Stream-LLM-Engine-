import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.format.core import (
        SSF_MAGIC,
        SSF_HEADER_SIZE,
        SSF_PAGE_SIZE,
        SSFVersion,
        TensorDType,
        _align_up,
        _sha256,
        _format_size,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestSSFConstants:
    def test_magic(self):
        assert SSF_MAGIC == b"SSF\x02"

    def test_header_size(self):
        assert SSF_HEADER_SIZE == 256

    def test_page_size(self):
        assert SSF_PAGE_SIZE == 4096


class TestSSFVersion:
    def test_enum_values(self):
        assert SSFVersion.V1 == 1
        assert SSFVersion.V2 == 2
        assert SSFVersion.V2_1 == 3


class TestTensorDType:
    def test_enum_values(self):
        assert TensorDType.F32 == 0
        assert TensorDType.F16 == 1
        assert TensorDType.BF16 == 2
        assert TensorDType.INT8 == 3

    def test_from_numpy_float32(self):
        result = TensorDType.from_numpy(np.float32)
        assert result == TensorDType.F32

    def test_from_numpy_float16(self):
        result = TensorDType.from_numpy(np.float16)
        assert result == TensorDType.F16

    def test_from_numpy_int8(self):
        result = TensorDType.from_numpy(np.int8)
        assert result == TensorDType.INT8

    def test_from_numpy_uint8(self):
        result = TensorDType.from_numpy(np.uint8)
        assert result == TensorDType.U8

    def test_from_numpy_unsupported(self):
        import pytest

        with pytest.raises(ValueError):
            TensorDType.from_numpy(np.int32)

    def test_to_numpy_f32(self):
        assert TensorDType.F32.to_numpy() == np.float32

    def test_to_numpy_f16(self):
        assert TensorDType.F16.to_numpy() == np.float16

    def test_to_numpy_int8(self):
        assert TensorDType.INT8.to_numpy() == np.int8

    def test_roundtrip(self):
        for dt in [np.float32, np.float16, np.int8, np.uint8]:
            tdt = TensorDType.from_numpy(dt)
            back = tdt.to_numpy()
            assert back == dt


class TestAlignUp:
    def test_align_up_exact(self):
        assert _align_up(4096, 4096) == 4096

    def test_align_up_need_padding(self):
        assert _align_up(1, 4096) == 4096
        assert _align_up(4097, 4096) == 8192

    def test_align_up_zero(self):
        assert _align_up(0, 4096) == 0

    def test_align_up_small_align(self):
        assert _align_up(5, 8) == 8
        assert _align_up(16, 8) == 16


class TestSha256:
    def test_sha256_deterministic(self):
        result1 = _sha256(b"test data")
        result2 = _sha256(b"test data")
        assert result1 == result2

    def test_sha256_length(self):
        result = _sha256(b"hello")
        assert len(result) == 32

    def test_sha256_empty(self):
        result = _sha256(b"")
        assert len(result) == 32


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500B"

    def test_kb(self):
        assert _format_size(2048) == "2.0KB"
        assert _format_size(1500) == "1.5KB"

    def test_mb(self):
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.0" in result

    def test_gb(self):
        result = _format_size(2 * 1024**3)
        assert "GB" in result
        assert "2.0" in result

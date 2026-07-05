from __future__ import annotations

import gzip
import hashlib
import json
import struct
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.format.core import (
        SSFVersion,
        TensorDType,
        SSF_MAGIC,
        SSF_HEADER_SIZE,
        SSF_FOOTER_SIZE,
        SSF_PAGE_SIZE,
        _align_up,
        _sha256,
        _format_size,
    )
    from spectralstream.format.header import SSFHeader
    from spectralstream.format.reader import SSFReader
    from spectralstream.format.writer import SSFWriter
except ImportError as e:
    pytest.skip(f"Import failed: {e}", allow_module_level=True)


class TestSSFVersion:
    def test_enum_values(self):
        assert SSFVersion.V1 == 1
        assert SSFVersion.V2 == 2
        assert SSFVersion.V2_1 == 3

    def test_is_int_enum(self):
        assert isinstance(SSFVersion.V1, int)
        assert int(SSFVersion.V2) == 2

    def test_all_members(self):
        members = {m.name: m.value for m in SSFVersion}
        assert members == {"V1": 1, "V2": 2, "V2_1": 3}


class TestTensorDType:
    def test_from_numpy_float32(self):
        assert TensorDType.from_numpy(np.dtype("float32")) == TensorDType.F32

    def test_from_numpy_float64(self):
        assert TensorDType.from_numpy(np.dtype("float64")) == TensorDType.F32

    def test_from_numpy_float16(self):
        assert TensorDType.from_numpy(np.dtype("float16")) == TensorDType.F16

    def test_from_numpy_int8(self):
        assert TensorDType.from_numpy(np.dtype("int8")) == TensorDType.INT8

    def test_from_numpy_uint8(self):
        assert TensorDType.from_numpy(np.dtype("uint8")) == TensorDType.U8

    def test_from_numpy_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            TensorDType.from_numpy(np.dtype("int32"))

    def test_from_numpy_unsupported_complex(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            TensorDType.from_numpy(np.dtype("complex64"))

    def test_to_numpy_f32(self):
        assert TensorDType.F32.to_numpy() == np.float32

    def test_to_numpy_f16(self):
        assert TensorDType.F16.to_numpy() == np.float16

    def test_to_numpy_int8(self):
        assert TensorDType.INT8.to_numpy() == np.int8

    def test_to_numpy_u8(self):
        assert TensorDType.U8.to_numpy() == np.uint8

    def test_to_numpy_bf16_fallback(self):
        result = TensorDType.BF16.to_numpy()
        try:
            bf16 = np.dtype("bfloat16")
            assert result == bf16
        except TypeError:
            assert result == np.float16

    def test_to_numpy_bf16_usable(self):
        result = TensorDType.BF16.to_numpy()
        arr = np.array([1.0, 2.0], dtype=result)
        assert arr.dtype == result

    def test_roundtrip_f32(self):
        dt = np.dtype("float32")
        assert TensorDType.from_numpy(dt).to_numpy() == dt

    def test_roundtrip_f16(self):
        dt = np.dtype("float16")
        assert TensorDType.from_numpy(dt).to_numpy() == dt

    def test_roundtrip_int8(self):
        dt = np.dtype("int8")
        assert TensorDType.from_numpy(dt).to_numpy() == dt

    def test_roundtrip_uint8(self):
        dt = np.dtype("uint8")
        assert TensorDType.from_numpy(dt).to_numpy() == dt

    def test_all_enum_values_unique(self):
        values = [m.value for m in TensorDType]
        assert len(values) == len(set(values))


class TestAlignUp:
    def test_no_align_needed(self):
        assert _align_up(16, 16) == 16

    def test_align_1(self):
        assert _align_up(0, 1) == 0
        assert _align_up(1, 1) == 1
        assert _align_up(100, 1) == 100

    def test_align_4(self):
        assert _align_up(0, 4) == 0
        assert _align_up(1, 4) == 4
        assert _align_up(3, 4) == 4
        assert _align_up(4, 4) == 4
        assert _align_up(5, 4) == 8

    def test_align_4096(self):
        assert _align_up(0, 4096) == 0
        assert _align_up(1, 4096) == 4096
        assert _align_up(4095, 4096) == 4096
        assert _align_up(4096, 4096) == 4096
        assert _align_up(4097, 4096) == 8192

    def test_large_values(self):
        assert _align_up(10**9, 4096) == ((10**9 + 4095) // 4096) * 4096

    def test_align_power_of_two(self):
        for align in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for val in [0, 1, align - 1, align, align + 1, align * 2 - 1]:
                expected = ((val + align - 1) // align) * align
                assert _align_up(val, align) == expected


class TestSHA256:
    def test_empty(self):
        assert _sha256(b"") == hashlib.sha256(b"").digest()

    def test_known_input(self):
        assert _sha256(b"hello") == hashlib.sha256(b"hello").digest()

    def test_large_input(self):
        data = b"x" * 100000
        assert _sha256(data) == hashlib.sha256(data).digest()

    def test_binary_data(self):
        data = bytes(range(256))
        assert _sha256(data) == hashlib.sha256(data).digest()

    def test_returns_32_bytes(self):
        assert len(_sha256(b"anything")) == 32


class TestFormatSize:
    def test_zero(self):
        assert _format_size(0) == "0B"

    def test_bytes(self):
        assert _format_size(1) == "1B"
        assert _format_size(512) == "512B"
        assert _format_size(1023) == "1023B"

    def test_kb_boundary(self):
        assert _format_size(1024) == "1.0KB"
        assert _format_size(1536) == "1.5KB"
        assert _format_size(1024 * 1024 - 1) == "1024.0KB"

    def test_mb_boundary(self):
        assert _format_size(1024**2) == "1.0MB"
        assert _format_size(int(1.5 * 1024**2)) == "1.5MB"
        assert _format_size(1024**3 - 1) == "1024.0MB"

    def test_gb_boundary(self):
        assert _format_size(1024**3) == "1.0GB"
        assert _format_size(int(2.5 * 1024**3)) == "2.5GB"

    def test_large_gb(self):
        assert _format_size(10 * 1024**3) == "10.0GB"

    def test_one_decimal_precision(self):
        result = _format_size(1024 * 1024 * 5)
        assert result.endswith("MB")
        assert "." in result


class TestConstants:
    def test_ssf_magic(self):
        assert SSF_MAGIC == b"SSF\x02"

    def test_ssf_header_size(self):
        assert SSF_HEADER_SIZE == 256

    def test_ssf_footer_size(self):
        assert SSF_FOOTER_SIZE == 128

    def test_ssf_page_size(self):
        assert SSF_PAGE_SIZE == 4096

    def test_constants_are_positive(self):
        assert SSF_HEADER_SIZE > 0
        assert SSF_FOOTER_SIZE > 0
        assert SSF_PAGE_SIZE > 0


class TestSSFHeader:
    def test_pack_size(self):
        h = SSFHeader()
        packed = h.pack()
        assert len(packed) == SSF_HEADER_SIZE

    def test_pack_starts_with_magic(self):
        h = SSFHeader()
        assert h.pack()[:4] == SSF_MAGIC

    def test_default_version(self):
        h = SSFHeader()
        assert h.version == 3

    def test_default_n_tensors(self):
        h = SSFHeader()
        assert h.n_tensors == 0

    def test_unpack_roundtrip_default(self):
        h = SSFHeader()
        packed = h.pack()
        h2 = SSFHeader.unpack(packed)
        assert h2.magic == h.magic
        assert h2.version == h.version
        assert h2.flags == h.flags
        assert h2.n_tensors == h.n_tensors
        assert h2.index_offset == h.index_offset
        assert h2.index_size == h.index_size
        assert h2.metadata_offset == h.metadata_offset
        assert h2.metadata_size == h.metadata_size
        assert h2.tensor_data_offset == h.tensor_data_offset
        assert h2.redundant_header_offset == h.redundant_header_offset
        assert h2.footer_offset == h.footer_offset

    def test_unpack_roundtrip_custom(self):
        h = SSFHeader(
            magic=SSF_MAGIC,
            version=2,
            flags=1,
            n_tensors=5,
            index_offset=4096,
            index_size=1024,
            metadata_offset=8192,
            metadata_size=512,
            tensor_data_offset=256,
            redundant_header_offset=4096,
            footer_offset=16384,
        )
        packed = h.pack()
        h2 = SSFHeader.unpack(packed)
        assert h2.magic == SSF_MAGIC
        assert h2.version == 2
        assert h2.flags == 1
        assert h2.n_tensors == 5
        assert h2.index_offset == 4096
        assert h2.index_size == 1024
        assert h2.metadata_offset == 8192
        assert h2.metadata_size == 512
        assert h2.tensor_data_offset == 256
        assert h2.redundant_header_offset == 4096
        assert h2.footer_offset == 16384

    def test_unpack_v3_accepted(self):
        h = SSFHeader(version=3)
        h2 = SSFHeader.unpack(h.pack())
        assert h2.version == 3

    def test_bad_magic_raises(self):
        packed = SSFHeader().pack()
        bad = b"\x00" * 4 + packed[4:]
        with pytest.raises(ValueError, match="Bad magic"):
            SSFHeader.unpack(bad)

    def test_unsupported_version_raises(self):
        h = SSFHeader(version=1)
        packed = h.pack()
        with pytest.raises(ValueError, match="Unsupported SSF version"):
            SSFHeader.unpack(packed)

    def test_too_short_data_raises(self):
        with pytest.raises(ValueError, match="Header too small"):
            SSFHeader.unpack(b"\x00" * 100)

    def test_trailing_bytes_ignored(self):
        h = SSFHeader(n_tensors=42)
        packed = h.pack() + b"trailing"
        h2 = SSFHeader.unpack(packed)
        assert h2.n_tensors == 42

    def test_pack_is_deterministic(self):
        h = SSFHeader(n_tensors=10, flags=3)
        assert h.pack() == h.pack()

    def test_reserved_bytes_are_zeros(self):
        h = SSFHeader()
        packed = h.pack()
        expected_size = struct.calcsize(SSFHeader.FORMAT)
        assert expected_size == SSF_HEADER_SIZE
        reserved_start = expected_size - 184
        assert packed[reserved_start:] == b"\x00" * 184


class TestSSFWriterReader:
    @pytest.fixture
    def temp_path(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        yield path
        import os

        try:
            os.unlink(path)
        except OSError:
            pass

    def test_write_and_read_simple_tensor(self, temp_path):
        tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        with SSFWriter(
            temp_path, metadata={"test": "value"}, compression_method=0
        ) as writer:
            info = writer.add_tensor("weights", tensor)
            assert info["name"] == "weights"
            assert info["shape"] == [2, 3]
            assert info["dtype"] == "F32"
            assert info["original_size"] > 0

        reader = SSFReader(temp_path)
        assert reader.header is not None
        assert reader.header.version == 3
        assert reader.header.n_tensors == 1
        assert reader.metadata["test"] == "value"
        assert reader.metadata["ssf_version"] == 3
        names = reader.tensor_names()
        assert names == ["weights"]

        info = reader.tensor_info("weights")
        assert info["name"] == "weights"
        assert info["shape"] == [2, 3]
        assert info["dtype"] == "F32"

        result = reader.get_tensor("weights")
        assert np.allclose(result, tensor)
        assert result.dtype == np.float32
        reader.close()

    def test_write_and_read_multiple_tensors(self, temp_path):
        t1 = np.array([1, 2, 3], dtype=np.int8)
        t2 = np.array([10, 20, 30, 40], dtype=np.uint8)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("a", t1)
            writer.add_tensor("b", t2)

        reader = SSFReader(temp_path)
        assert reader.header.n_tensors == 2
        assert set(reader.tensor_names()) == {"a", "b"}
        assert np.array_equal(reader.get_tensor("a"), t1)
        assert np.array_equal(reader.get_tensor("b"), t2)
        reader.close()

    def test_reader_getitem(self, temp_path):
        t = np.array([1.5, 2.5], dtype=np.float32)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("x", t)
        reader = SSFReader(temp_path)
        assert np.allclose(reader["x"], t)
        reader.close()

    def test_reader_iteration(self, temp_path):
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("a", np.array([1], dtype=np.float32))
            writer.add_tensor("b", np.array([2], dtype=np.float32))
        reader = SSFReader(temp_path)
        names = set()
        for name, tensor in reader:
            names.add(name)
            assert tensor.shape == (1,)
        assert names == {"a", "b"}
        reader.close()

    def test_reader_len(self, temp_path):
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("a", np.array([1], dtype=np.float32))
        reader = SSFReader(temp_path)
        assert len(reader) == 1
        reader.close()

    def test_reader_empty_file_returns_none_header(self, temp_path):
        open(temp_path, "w").close()
        reader = SSFReader(temp_path)
        assert reader.header is None
        reader.close()

    def test_reader_tensor_not_found_raises(self, temp_path):
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("a", np.array([1], dtype=np.float32))
        reader = SSFReader(temp_path)
        with pytest.raises(KeyError, match="not found"):
            reader.get_tensor("nonexistent")
        reader.close()

    def test_reader_list_tensors(self, temp_path):
        tensor = np.array([[1, 2], [3, 4]], dtype=np.int8)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("m", tensor)
        reader = SSFReader(temp_path)
        tensors = reader.list_tensors()
        assert len(tensors) == 1
        assert tensors[0]["name"] == "m"
        assert tensors[0]["shape"] == [2, 2]
        assert tensors[0]["dtype"] == "INT8"
        reader.close()

    def test_write_with_lossless_zlib(self, temp_path):
        tensor = np.random.randn(100).astype(np.float32)
        with SSFWriter(temp_path, compression_method=350) as writer:
            writer.add_tensor("data", tensor)
        reader = SSFReader(temp_path)
        result = reader.get_tensor("data")
        assert np.allclose(result, tensor, atol=1e-5)
        reader.close()

    def test_reader_verify_valid_file(self, temp_path):
        tensor = np.array([1.0], dtype=np.float32)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("t", tensor)
        reader = SSFReader(temp_path)
        report = reader.verify()
        assert report["valid"] is True
        assert report["header_ok"] is True
        assert report["index_ok"] is True
        assert report["tensor_checksums"].get("t") == "ok"
        reader.close()

    def test_write_with_metadata(self, temp_path):
        meta = {"author": "test", "version": "1.0", "model": "test_model"}
        with SSFWriter(temp_path, metadata=meta, compression_method=0) as writer:
            writer.add_tensor("t", np.array([1], dtype=np.float32))
        reader = SSFReader(temp_path)
        for k, v in meta.items():
            assert reader.metadata[k] == v
        reader.close()

    def test_roundtrip_f16(self, temp_path):
        tensor = np.array([1.5, 2.5, 3.5], dtype=np.float16)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("t", tensor)
        reader = SSFReader(temp_path)
        result = reader.get_tensor("t")
        assert result.dtype == np.float16
        assert np.allclose(result, tensor)
        reader.close()

    def test_roundtrip_int8(self, temp_path):
        tensor = np.array([-128, 0, 127], dtype=np.int8)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("t", tensor)
        reader = SSFReader(temp_path)
        result = reader.get_tensor("t")
        assert np.array_equal(result, tensor)
        reader.close()

    def test_roundtrip_uint8(self, temp_path):
        tensor = np.array([0, 128, 255], dtype=np.uint8)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("t", tensor)
        reader = SSFReader(temp_path)
        result = reader.get_tensor("t")
        assert np.array_equal(result, tensor)
        reader.close()

    def test_get_tensors_batch(self, temp_path):
        a = np.array([1, 2], dtype=np.float32)
        b = np.array([3, 4, 5], dtype=np.float32)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("a", a)
            writer.add_tensor("b", b)
        reader = SSFReader(temp_path)
        result = reader.get_tensors(["a", "b"])
        assert "a" in result and "b" in result
        assert np.allclose(result["a"], a)
        assert np.allclose(result["b"], b)
        reader.close()

    def test_reader_cache_eviction(self, temp_path):
        tensors = {f"t{i}": np.array([i], dtype=np.float32) for i in range(40)}
        with SSFWriter(temp_path, compression_method=0) as writer:
            for name, t in tensors.items():
                writer.add_tensor(name, t)
        reader = SSFReader(temp_path, cache_size=16)
        for name in tensors:
            reader.get_tensor(name)
        # internal cache should be bounded by cache_size
        with reader._lock:
            assert len(reader._cache) <= 16
        reader.close()

    def test_extract_subset(self, temp_path):
        a = np.array([1.0], dtype=np.float32)
        b = np.array([2.0], dtype=np.float32)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("a", a)
            writer.add_tensor("b", b)
        reader = SSFReader(temp_path)
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            out_path = f.name
        try:
            result = reader.extract_subset(["a"], out_path, metadata={"subset": "yes"})
            assert result["n_tensors"] == 1
            r2 = SSFReader(out_path)
            assert "a" in r2.tensor_names()
            assert "b" not in r2.tensor_names()
            assert np.allclose(r2.get_tensor("a"), a)
            assert r2.metadata.get("subset") == "yes"
            r2.close()
        finally:
            import os

            try:
                os.unlink(out_path)
            except OSError:
                pass
        reader.close()

    def test_reader_get_quality_report(self, temp_path):
        tensor = np.random.randn(50).astype(np.float32)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("data", tensor)
        reader = SSFReader(temp_path)
        report = reader.get_quality_report()
        assert "tensors" in report
        assert "aggregate" in report
        assert report["aggregate"]["n_tensors"] == 1
        assert report["tensors"][0]["name"] == "data"
        reader.close()

    def test_context_manager_writer(self, temp_path):
        t = np.array([1.0], dtype=np.float32)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("x", t)
        with SSFReader(temp_path) as reader:
            assert np.allclose(reader["x"], t)

    def test_reader_read_tensor_chunk_2d(self, temp_path):
        t = np.arange(20, dtype=np.float32).reshape(5, 4)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("m", t)
        reader = SSFReader(temp_path)
        chunk = reader.read_tensor_chunk("m", row_start=1, row_end=3)
        assert chunk.shape == (2, 4)
        assert np.allclose(chunk, t[1:3])
        reader.close()

    def test_reader_with_mmap_disabled(self, temp_path):
        t = np.array([1.0], dtype=np.float32)
        with SSFWriter(temp_path, compression_method=0) as writer:
            writer.add_tensor("x", t)
        reader = SSFReader(temp_path, mmap_mode=False)
        assert np.allclose(reader["x"], t)
        reader.close()

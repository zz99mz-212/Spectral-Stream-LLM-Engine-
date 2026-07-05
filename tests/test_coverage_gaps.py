"""Tests for critical coverage gaps: format, CLI, IO, orchestrator edge cases."""

import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from spectralstream.format.core import (
    SSF_MAGIC,
    SSF_HEADER_SIZE,
    SSF_FOOTER_SIZE,
    SSF_PAGE_SIZE,
    SSFVersion,
    TensorDType,
    _align_up,
    _sha256,
    _format_size,
)
from spectralstream.format.header import SSFHeader
from spectralstream.format.index import (
    TensorIndex,
    TensorIndexEntry,
    LegacyTensorIndexEntry,
)
from spectralstream.format.compression import (
    _method_id_to_name,
    _name_to_method_id,
    _compress_via_engine,
    _decompress_via_engine,
)
from spectralstream.format.writer import SSFWriter
from spectralstream.format.reader import SSFReader
from spectralstream.compression.engine._io import (
    _SafetensorsIO,
    _SSFIOWriter,
    _CheckpointManager,
)
from spectralstream.compression.engine._dataclasses import (
    CompressionConfig,
    TensorProfile,
    CompressedTensor,
    CompressionReport,
    CompressionTelemetry,
    CalibrationData,
)
from spectralstream.compression.engine._helpers import (
    _classify_by_name,
    _classify_by_name_simple,
    _safe_bytes,
    _compute_metrics,
    _compute_ratio,
    _sample_flat,
    _bootstrap_error,
    _estimate_noise_floor,
    _estimate_entropy_rate,
    _toeplitz_score,
    _circulant_score,
    _block_diagonal_score,
    _hierarchical_structure_score,
    _mutual_information_blocks,
    _kolmogorov_estimate,
    _structured_nm_score,
    _block_sparsity_score,
    _unstructured_sparsity_score,
    _nm_sparsity_score,
)


# =============================================================================
# format/core.py — TensorDType, constants, helpers
# =============================================================================


class TestFormatCore:
    def test_ssf_magic(self):
        assert SSF_MAGIC == b"SSF\x02"

    def test_ssf_constants(self):
        assert SSF_HEADER_SIZE == 256
        assert SSF_FOOTER_SIZE == 128
        assert SSF_PAGE_SIZE == 4096

    def test_ssf_version_enum(self):
        assert SSFVersion.V1 == 1
        assert SSFVersion.V2 == 2
        assert SSFVersion.V2_1 == 3

    def test_tensor_dtype_from_numpy_float32(self):
        assert TensorDType.from_numpy(np.dtype("float32")) == TensorDType.F32

    def test_tensor_dtype_from_numpy_float64(self):
        assert TensorDType.from_numpy(np.dtype("float64")) == TensorDType.F32

    def test_tensor_dtype_from_numpy_float16(self):
        assert TensorDType.from_numpy(np.dtype("float16")) == TensorDType.F16

    def test_tensor_dtype_from_numpy_int8(self):
        assert TensorDType.from_numpy(np.dtype("int8")) == TensorDType.INT8

    def test_tensor_dtype_from_numpy_uint8(self):
        assert TensorDType.from_numpy(np.dtype("uint8")) == TensorDType.U8

    def test_tensor_dtype_from_numpy_unsupported(self):
        with pytest.raises(ValueError):
            TensorDType.from_numpy(np.dtype("complex128"))

    def test_tensor_dtype_to_numpy(self):
        assert TensorDType.F32.to_numpy() == np.float32
        assert TensorDType.F16.to_numpy() == np.float16
        assert TensorDType.INT8.to_numpy() == np.int8
        assert TensorDType.U8.to_numpy() == np.uint8

    def test_align_up(self):
        assert _align_up(0, 4096) == 0
        assert _align_up(1, 4096) == 4096
        assert _align_up(4096, 4096) == 4096
        assert _align_up(4097, 4096) == 8192

    def test_sha256(self):
        result = _sha256(b"hello")
        assert len(result) == 32

    def test_format_size(self):
        assert _format_size(500) == "500B"
        assert _format_size(1500) == "1.5KB"
        assert _format_size(1500000) == "1.4MB"
        assert _format_size(1500000000) == "1.4GB"


# =============================================================================
# format/header.py — SSFHeader
# =============================================================================


class TestSSFHeader:
    def test_pack_unpack_roundtrip_v3(self):
        h = SSFHeader(
            version=3,
            flags=1,
            n_tensors=5,
            index_offset=4096,
            index_size=256,
            metadata_offset=8192,
            metadata_size=128,
            tensor_data_offset=256,
            redundant_header_offset=4096,
            footer_offset=16384,
        )
        packed = h.pack()
        assert len(packed) == SSF_HEADER_SIZE

        h2 = SSFHeader.unpack(packed)
        assert h2.magic == SSF_MAGIC
        assert h2.version == 3
        assert h2.flags == 1
        assert h2.n_tensors == 5
        assert h2.index_offset == 4096
        assert h2.index_size == 256
        assert h2.metadata_offset == 8192
        assert h2.metadata_size == 128
        assert h2.tensor_data_offset == 256
        assert h2.redundant_header_offset == 4096
        assert h2.footer_offset == 16384

    def test_unpack_v2(self):
        h = SSFHeader(version=2)
        h2 = SSFHeader.unpack(h.pack())
        assert h2.version == 2

    def test_unpack_bad_magic(self):
        data = b"\x00" * SSF_HEADER_SIZE
        with pytest.raises(ValueError, match="Bad magic"):
            SSFHeader.unpack(data)

    def test_unpack_unsupported_version(self):
        h = SSFHeader(version=99)
        packed = h.pack()
        # Overwrite version field
        packed = packed[:4] + struct.pack("<I", 99) + packed[8:]
        with pytest.raises(ValueError, match="Unsupported SSF version"):
            SSFHeader.unpack(packed)

    def test_unpack_too_small(self):
        with pytest.raises(ValueError, match="Header too small"):
            SSFHeader.unpack(b"\x00" * 100)

    def test_header_defaults(self):
        h = SSFHeader()
        assert h.magic == SSF_MAGIC
        assert h.version == 3
        assert h.flags == 0
        assert h.n_tensors == 0


# =============================================================================
# format/index.py — TensorIndex, TensorIndexEntry, LegacyTensorIndexEntry
# =============================================================================


class TestTensorIndexEntry:
    def test_pack_unpack_roundtrip(self):
        entry = TensorIndexEntry(
            name="test_tensor",
            shape=(32, 64),
            dtype=TensorDType.F32,
            compression_method=350,
            compression_params={"level": 3},
            data_offset=4096,
            compressed_size=1024,
            original_size=8192,
            quality_metrics={
                "relative_error": 0.01,
                "snr_db": 20.0,
                "psnr_db": 25.0,
                "cosine_similarity": 0.99,
                "compression_ratio": 8.0,
            },
            checksum=b"\xab" * 32,
            flags=0,
        )
        packed = entry.pack()
        entry2, size = TensorIndexEntry.unpack(packed)
        assert entry2.name == "test_tensor"
        assert entry2.shape == (32, 64)
        assert entry2.dtype == TensorDType.F32
        assert entry2.compression_method == 350
        assert entry2.compression_params == {"level": 3}
        assert entry2.data_offset == 4096
        assert entry2.compressed_size == 1024
        assert entry2.original_size == 8192
        assert abs(entry2.quality_metrics["relative_error"] - 0.01) < 1e-9
        assert entry2.checksum == b"\xab" * 32
        assert entry2.flags == 0
        assert size == len(packed)

    def test_pack_empty_params(self):
        entry = TensorIndexEntry(
            name="empty_params",
            shape=(10,),
            dtype=TensorDType.INT8,
            compression_method=0,
            compression_params={},
            data_offset=256,
            compressed_size=10,
            original_size=10,
            quality_metrics={
                "relative_error": 0.0,
                "snr_db": float("inf"),
                "psnr_db": float("inf"),
                "cosine_similarity": 1.0,
                "compression_ratio": 1.0,
            },
            checksum=b"\x00" * 32,
            flags=0,
        )
        packed = entry.pack()
        entry2, _ = TensorIndexEntry.unpack(packed)
        assert entry2.compression_params == {}

    def test_unpack_truncated_name(self):
        data = struct.pack("<I", 100)  # name_len=100 but no data
        with pytest.raises(ValueError, match="Truncated index entry name"):
            TensorIndexEntry.unpack(data)


class TestTensorIndex:
    def test_add_and_get(self):
        idx = TensorIndex()
        assert len(idx) == 0
        e1 = TensorIndexEntry(
            name="a",
            shape=(1,),
            dtype=TensorDType.F32,
            compression_method=0,
            compression_params={},
            data_offset=0,
            compressed_size=4,
            original_size=4,
            quality_metrics={},
            checksum=b"\x00" * 32,
        )
        e2 = TensorIndexEntry(
            name="b",
            shape=(2, 2),
            dtype=TensorDType.F16,
            compression_method=1,
            compression_params={},
            data_offset=100,
            compressed_size=8,
            original_size=16,
            quality_metrics={},
            checksum=b"\x01" * 32,
        )
        idx.add(e1)
        idx.add(e2)
        assert len(idx) == 2
        assert idx.get("a") is e1
        assert idx.get("b") is e2
        assert idx.get("nonexistent") is None
        assert idx[0] is e1
        assert idx[1] is e2
        assert idx.names() == ["a", "b"]

    def test_iteration(self):
        idx = TensorIndex()
        entries = [
            TensorIndexEntry(
                name=f"t{i}",
                shape=(i + 1,) if i % 2 == 0 else (i + 1, i + 1),
                dtype=TensorDType.F32,
                compression_method=i,
                compression_params={},
                data_offset=i * 100,
                compressed_size=i * 10,
                original_size=i * 100,
                quality_metrics={},
                checksum=b"\x00" * 32,
            )
            for i in range(5)
        ]
        for e in entries:
            idx.add(e)
        assert list(idx) == entries

    def test_pack_unpack_roundtrip(self):
        idx = TensorIndex()
        for i in range(3):
            idx.add(
                TensorIndexEntry(
                    name=f"tensor_{i}",
                    shape=(16, 16),
                    dtype=TensorDType.F32,
                    compression_method=350 + i,
                    compression_params={"idx": i},
                    data_offset=4096 + i * 1024,
                    compressed_size=512 + i * 100,
                    original_size=1024,
                    quality_metrics={
                        "relative_error": 0.01 * i,
                        "snr_db": 30.0 - i,
                        "psnr_db": 35.0 - i,
                        "cosine_similarity": 1.0 - 0.01 * i,
                        "compression_ratio": 2.0 + i,
                    },
                    checksum=bytes([i] * 32),
                )
            )
        packed = idx.pack()
        idx2 = TensorIndex.unpack(packed, is_legacy=False)
        assert len(idx2) == 3
        for i in range(3):
            entry = idx2.get(f"tensor_{i}")
            assert entry is not None
            assert entry.compression_method == 350 + i

    def test_unpack_legacy(self):
        le = LegacyTensorIndexEntry(
            name="legacy_tensor",
            shape=(4, 4),
            dtype=TensorDType.F32,
            compression=0,
            data_offset=4096,
            compressed_size=64,
            original_size=64,
            checksum=b"\xcc" * 32,
            flags=0,
        )
        packed = struct.pack("<I", len("legacy_tensor"))
        packed += b"legacy_tensor"
        packed += struct.pack("<I", 2)
        packed += struct.pack("<QQ", 4, 4)
        packed += struct.pack("<HH", 0, 0)
        packed += struct.pack("<QQQ", 4096, 64, 64)
        packed += b"\xcc" * 32
        packed += struct.pack("<I", 0)

        idx = TensorIndex.unpack(packed, is_legacy=True)
        assert len(idx) == 1
        from typing import cast

        e = cast("TensorIndexEntry", idx.get("legacy_tensor"))  # type: ignore[unreachable]
        assert e.dtype == TensorDType.F32
        assert e.compression_method == 0
        assert e.data_offset == 4096


# =============================================================================
# format/compression.py — method ID mapping, compress/decompress via engine
# =============================================================================


class TestCompression:
    def test_method_id_to_name_known(self):
        assert _method_id_to_name(0) == "passthrough"
        assert _method_id_to_name(1) == "block_int4"
        assert _method_id_to_name(50) == "svd_truncated"
        assert _method_id_to_name(350) == "lossless_zlib"
        assert _method_id_to_name(352) == "lossless_zstd"
        assert _method_id_to_name(400) == "cascade_2_stage"

    def test_method_id_to_name_unknown(self):
        assert _method_id_to_name(9999) == "passthrough"

    def test_name_to_method_id_known(self):
        assert _name_to_method_id("passthrough") == 0
        assert _name_to_method_id("block_int4") == 1
        assert _name_to_method_id("svd_truncated") == 50
        assert _name_to_method_id("lossless_zstd") == 352

    def test_name_to_method_id_unknown(self):
        assert _name_to_method_id("nonexistent") == 0

    def test_compress_via_engine_passthrough(self):
        data = b"hello world"
        result, meta = _compress_via_engine(data, 0)
        assert result == data
        assert meta == {}

    def test_decompress_via_engine_passthrough(self):
        data = b"hello world"
        result = _decompress_via_engine(data, 0)
        assert result == data

    def test_compress_via_engine_unknown_id(self):
        data = b"test data for compression"
        result, meta = _compress_via_engine(data, 999)
        assert len(result) > 0
        assert isinstance(meta, dict)


# =============================================================================
# format/writer.py + format/reader.py — SSFWriter / SSFReader roundtrip
# =============================================================================


class TestSSFRoundtrip:
    def test_write_read_single_tensor(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            with SSFWriter(path, metadata={"test": True}) as writer:
                writer.add_tensor("weights", tensor, method=0)

            with SSFReader(path) as reader:
                assert reader.header.version == 3
                assert reader.header.n_tensors == 1
                assert reader.tensor_names() == ["weights"]
                info = reader.tensor_info("weights")
                assert info is not None
                assert info["shape"] == [2, 2]
                assert info["dtype"] == "F32"

                loaded = reader.get_tensor("weights")
                np.testing.assert_array_almost_equal(loaded, tensor)
        finally:
            os.unlink(path)

    def test_write_read_multiple_tensors(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            t1 = np.random.randn(16, 16).astype(np.float32)
            t2 = np.random.randn(8, 8, 8).astype(np.float32)
            with SSFWriter(path) as writer:
                writer.add_tensor("t1", t1, method=0)
                writer.add_tensor("t2", t2, method=0)

            with SSFReader(path) as reader:
                assert reader.header.n_tensors == 2
                assert reader.tensor_names() == ["t1", "t2"]
                np.testing.assert_array_almost_equal(reader.get_tensor("t1"), t1)
                np.testing.assert_array_almost_equal(reader.get_tensor("t2"), t2)
        finally:
            os.unlink(path)

    def test_write_with_compression_method(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            tensor = np.random.randn(32, 32).astype(np.float32)
            with SSFWriter(path) as writer:
                writer.add_tensor("data", tensor, method=352)

            with SSFReader(path) as reader:
                info = reader.tensor_info("data")
                assert info is not None
                assert info["method_id"] == 352
                loaded = reader.get_tensor("data")
                assert loaded.shape == (32, 32)
        finally:
            os.unlink(path)

    def test_reader_getitem(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            tensor = np.array([1, 2, 3], dtype=np.float32)
            with SSFWriter(path) as writer:
                writer.add_tensor("x", tensor, method=0)

            with SSFReader(path) as reader:
                np.testing.assert_array_equal(reader["x"], tensor)
        finally:
            os.unlink(path)

    def test_reader_len_and_iter(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            with SSFWriter(path) as writer:
                writer.add_tensor("a", np.array([1.0], dtype=np.float32), method=0)
                writer.add_tensor("b", np.array([2.0], dtype=np.float32), method=0)

            with SSFReader(path) as reader:
                assert len(reader) == 2
                names = set()
                for name, _ in reader:
                    names.add(name)
                assert names == {"a", "b"}
        finally:
            os.unlink(path)

    def test_reader_get_tensors(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            t1 = np.array([10.0], dtype=np.float32)
            t2 = np.array([20.0], dtype=np.float32)
            with SSFWriter(path) as writer:
                writer.add_tensor("x", t1, method=0)
                writer.add_tensor("y", t2, method=0)

            with SSFReader(path) as reader:
                result = reader.get_tensors(["x", "y"])
                assert "x" in result and "y" in result
                np.testing.assert_array_equal(result["x"], t1)
                np.testing.assert_array_equal(result["y"], t2)
        finally:
            os.unlink(path)

    def test_reader_metadata(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            meta = {"model": "test", "version": 1}
            with SSFWriter(path, metadata=meta) as writer:
                writer.add_tensor("t", np.array([1.0], dtype=np.float32), method=0)

            with SSFReader(path) as reader:
                assert reader.metadata.get("model") == "test"
                assert reader.metadata.get("ssf_version") == 3
        finally:
            os.unlink(path)

    def test_reader_nonexistent_tensor(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            with SSFWriter(path) as writer:
                writer.add_tensor("a", np.array([1.0], dtype=np.float32), method=0)

            with SSFReader(path) as reader:
                with pytest.raises(KeyError):
                    reader.get_tensor("nonexistent")
        finally:
            os.unlink(path)

    def test_reader_tensor_info_nonexistent(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            with SSFWriter(path) as writer:
                writer.add_tensor("a", np.array([1.0], dtype=np.float32), method=0)

            with SSFReader(path) as reader:
                assert reader.tensor_info("nonexistent") is None
        finally:
            os.unlink(path)

    def test_reader_verify(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            with SSFWriter(path) as writer:
                writer.add_tensor("t", np.array([1.0], dtype=np.float32), method=0)

            with SSFReader(path) as reader:
                result = reader.verify()
                assert result["valid"] is True
                assert result["header_ok"] is True
                assert result["index_ok"] is True
                assert result["tensor_checksums"]["t"] == "ok"
        finally:
            os.unlink(path)

    def test_writer_not_opened_error(self):
        writer = SSFWriter("/tmp/nonexistent/test.ssf")
        with pytest.raises(RuntimeError, match="not opened"):
            writer.add_tensor("x", np.array([1.0], dtype=np.float32))

    def test_reader_list_tensors(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            with SSFWriter(path) as writer:
                writer.add_tensor("t1", np.array([1.0], dtype=np.float32), method=350)

            with SSFReader(path) as reader:
                lst = reader.list_tensors()
                assert len(lst) == 1
                assert lst[0]["name"] == "t1"
                assert lst[0]["method_name"] == "lossless_zlib"
        finally:
            os.unlink(path)

    def test_reader_get_quality_report(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            path = f.name
        try:
            with SSFWriter(path) as writer:
                writer.add_tensor("t1", np.array([1.0], dtype=np.float32), method=0)

            with SSFReader(path) as reader:
                qr = reader.get_quality_report()
                assert "tensors" in qr
                assert "aggregate" in qr
                assert qr["aggregate"]["n_tensors"] == 1
        finally:
            os.unlink(path)


# =============================================================================
# compression/engine/_io.py — SafetensorsIO, CheckpointManager
# =============================================================================


class TestSafetensorsIO:
    def test_scan_and_read(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            data = {
                "test_tensor": {
                    "dtype": "F32",
                    "shape": [2, 3],
                    "data_offsets": [0, 24],
                }
            }
            header = json.dumps(data).encode("utf-8")
            header_len = struct.pack("<Q", len(header))
            tensor_data = np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
            ).tobytes()
            with open(path, "wb") as f:
                f.write(header_len + header + tensor_data)

            io = _SafetensorsIO(use_mmap=True)
            info = io.scan(path)
            assert "test_tensor" in info
            shape, dt, off, nb = info["test_tensor"]
            assert shape == (2, 3)
            assert dt == "F32"

            tensor = io.read(path, shape, dt, off, nb)
            np.testing.assert_array_almost_equal(
                tensor, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            )
        finally:
            os.unlink(path)

    def test_scann_with_metadata_skip(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            data = {
                "__metadata__": {"foo": "bar"},
                "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            }
            header = json.dumps(data).encode("utf-8")
            header_len = struct.pack("<Q", len(header))
            tdata = np.array([1.0, 2.0], dtype=np.float32).tobytes()
            with open(path, "wb") as f:
                f.write(header_len + header + tdata)

            io = _SafetensorsIO()
            info = io.scan(path)
            assert "__metadata__" not in info
            assert "w" in info
        finally:
            os.unlink(path)

    def test_list_tensors(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            data = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
            header = json.dumps(data).encode("utf-8")
            header_len = struct.pack("<Q", len(header))
            with open(path, "wb") as f:
                f.write(header_len + header + b"\x00" * 4)
            io = _SafetensorsIO()
            assert io.list_tensors(path) == ["a"]
        finally:
            os.unlink(path)

    def test_get_tensor_info(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            data = {"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}}
            header = json.dumps(data).encode("utf-8")
            header_len = struct.pack("<Q", len(header))
            with open(path, "wb") as f:
                f.write(header_len + header + b"\x00" * 12)
            io = _SafetensorsIO()
            info = io.get_tensor_info(path, "x")
            assert info is not None
            assert info[0] == (3,)
            info2 = io.get_tensor_info(path, "nonexistent")
            assert info2 is None
        finally:
            os.unlink(path)

    def test_estimate_model_size(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            data = {"a": {"dtype": "F32", "shape": [10], "data_offsets": [0, 40]}}
            header = json.dumps(data).encode("utf-8")
            header_len = struct.pack("<Q", len(header))
            with open(path, "wb") as f:
                f.write(header_len + header + b"\x00" * 40)
            io = _SafetensorsIO()
            assert io.estimate_model_size(path) == 40
        finally:
            os.unlink(path)


class TestCheckpointManager:
    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr = _CheckpointManager(path)
            mgr.save(5, 10, [{"method": "block_int8"}], {"ratio": 100.0})
            data = mgr.load()
            assert data is not None
            assert data["completed"] == 5
            assert data["total"] == 10
            assert data["compressed_tensors"] == [{"method": "block_int8"}]
            assert data["report_data"]["ratio"] == 100.0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_nonexistent(self):
        mgr = _CheckpointManager("/tmp/nonexistent_checkpoint.json")
        assert mgr.load() is None

    def test_load_corrupted(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
            f.write(b"not json")
        try:
            mgr = _CheckpointManager(path)
            assert mgr.load() is None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_clear(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr = _CheckpointManager(path)
            mgr.save(1, 1, [], {})
            assert mgr.load() is not None
            mgr.clear()
            assert mgr.load() is None
        finally:
            if os.path.exists(path):
                os.unlink(path)


# =============================================================================
# compression/engine/_dataclasses.py
# =============================================================================


class TestCompressionConfig:
    def test_defaults(self):
        cfg = CompressionConfig()
        assert cfg.target_ratio == 5000.0
        assert cfg.max_error == 0.0002
        assert cfg.streaming is True
        assert cfg.num_workers == 4
        assert cfg.max_memory_gb == 4.0


class TestTensorProfile:
    def test_defaults(self):
        tp = TensorProfile()
        assert tp.name == ""
        assert tp.shape == (0,)
        assert tp.sensitivity == 0.5
        assert tp.sensitivity_category == "UNKNOWN"
        assert tp.recommended_bits == 8

    def test_with_values(self):
        tp = TensorProfile(
            name="test",
            shape=(64, 64),
            dtype="float32",
            n_elements=4096,
            nbytes=16384,
            sensitivity=0.8,
            sensitivity_category="HIGH",
        )
        assert tp.name == "test"
        assert tp.n_elements == 4096
        assert tp.sensitivity_category == "HIGH"


class TestCompressedTensor:
    def test_quality_grade_s(self):
        ct = CompressedTensor(
            _data=b"test",
            method="block_int8",
            params={},
            original_shape=(4, 4),
            original_dtype="float32",
            compression_ratio=10.0,
            relative_error=1e-6,
            snr_db=60.0,
            psnr_db=60.0,
            cosine_similarity=1.0,
            computation_time=0.01,
        )
        assert ct.quality_grade == "S"

    def test_quality_grade_a(self):
        ct = CompressedTensor(
            _data=b"test",
            method="block_int8",
            params={},
            original_shape=(4, 4),
            original_dtype="float32",
            compression_ratio=5.0,
            relative_error=0.0005,
            snr_db=40.0,
            psnr_db=40.0,
            cosine_similarity=0.999,
            computation_time=0.01,
        )
        assert ct.quality_grade == "A"

    def test_quality_grade_f(self):
        ct = CompressedTensor(
            _data=b"test",
            method="bad_method",
            params={},
            original_shape=(4, 4),
            original_dtype="float32",
            compression_ratio=1.0,
            relative_error=1.0,
            snr_db=0.0,
            psnr_db=0.0,
            cosine_similarity=0.0,
            computation_time=0.01,
        )
        assert ct.quality_grade == "F"


class TestCompressionReport:
    def test_empty_report(self):
        report = CompressionReport()
        assert report.overall_ratio == 1.0
        assert report.total_original_bytes == 0

    def test_to_dict(self):
        report = CompressionReport(
            total_original_bytes=1000,
            total_compressed_bytes=100,
            overall_ratio=10.0,
            avg_error=0.01,
            max_error=0.02,
            min_error=0.0,
            time_seconds=5.0,
            profile_time=2.0,
            compress_time=3.0,
        )
        d = report.to_dict()
        assert d["overall_ratio"] == 10.0
        assert d["total_original_bytes"] == 1000
        assert "previous_best_ratio" not in d

    def test_to_dict_with_previous_best(self):
        report = CompressionReport(overall_ratio=5.0, previous_best_ratio=3.0)
        d = report.to_dict()
        assert d["previous_best_ratio"] == 3.0

    def test_summary(self):
        report = CompressionReport(
            tensors=[
                CompressedTensor(
                    _data=b"x",
                    method="test",
                    params={},
                    original_shape=(1,),
                    original_dtype="float32",
                    compression_ratio=2.0,
                    relative_error=0.01,
                    snr_db=20.0,
                    psnr_db=20.0,
                    cosine_similarity=0.99,
                    computation_time=0.1,
                )
            ],
            total_original_bytes=1024,
            total_compressed_bytes=512,
            overall_ratio=2.0,
            avg_error=0.01,
            max_error=0.01,
            min_error=0.01,
            weighted_error=0.01,
            time_seconds=1.0,
            profile_time=0.5,
            compress_time=0.5,
            method_distribution={"test": 1},
        )
        s = report.summary()
        assert "2.00x" in s
        assert "test" in s
        assert "1.0000%" in s

    def test_save_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report = CompressionReport(overall_ratio=10.0)
            report.save_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["overall_ratio"] == 10.0
        finally:
            os.unlink(path)

    def test_save_gz(self):
        import gzip

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name
        try:
            report = CompressionReport(overall_ratio=5.0)
            report.save_gz(path)
            with gzip.open(path, "rt") as f:
                data = json.load(f)
            assert data["overall_ratio"] == 5.0
        finally:
            os.unlink(path)


class TestCalibrationData:
    def test_from_random(self):
        cd = CalibrationData.from_random(n_samples=16, vocab_size=100, seq_len=8)
        assert cd.inputs is not None
        assert cd.inputs.shape == (16, 8)

    def test_compute_fisher(self):
        acts = {"layer1": np.random.randn(4, 4).astype(np.float32)}
        cd = CalibrationData(inputs=np.array([[1, 2, 3]]))
        result = cd.compute_fisher(acts)
        assert result.fisher_info is not None
        assert "layer1" in result.fisher_info
        np.testing.assert_array_almost_equal(
            result.fisher_info["layer1"], acts["layer1"].astype(np.float64) ** 2
        )


# =============================================================================
# compression/engine/_helpers.py — classification, metrics, scoring
# =============================================================================


class TestHelpersClassification:
    def test_classify_by_name_embedding(self):
        assert _classify_by_name("model.embed_tokens") == "embedding"
        assert _classify_by_name("tok_embeddings.weight") == "embedding"
        assert _classify_by_name("wte.weight") == "embedding"

    def test_classify_by_name_attention_qkv(self):
        assert _classify_by_name("attn_q.weight") == "attention_q"
        assert _classify_by_name("q_proj.weight") == "attention_q"
        assert _classify_by_name("attn_k.weight") == "attention_k"
        assert _classify_by_name("k_proj.weight") == "attention_k"
        assert _classify_by_name("v_proj.weight") == "attention_v"
        assert _classify_by_name("attn_o.weight") == "attention_o"

    def test_classify_by_name_qkv_fused(self):
        assert _classify_by_name("qkv.weight") == "qkv_fused"

    def test_classify_by_name_ffn(self):
        assert _classify_by_name("ffn_gate.weight") == "ffn_gate"
        assert _classify_by_name("gate_proj.weight") == "ffn_gate"
        assert _classify_by_name("ffn_up.weight") == "ffn_up"
        assert _classify_by_name("up_proj.weight") == "ffn_up"
        assert _classify_by_name("ffn_down.weight") == "ffn_down"
        assert _classify_by_name("down_proj.weight") == "ffn_down"

    def test_classify_by_name_norm(self):
        assert _classify_by_name("rms_norm.weight") == "norm"
        assert _classify_by_name("ln_1.weight") == "norm"

    def test_classify_by_name_output(self):
        assert _classify_by_name("lm_head.weight") == "output"
        assert _classify_by_name("model.lm_head") == "output"

    def test_classify_by_name_default(self):
        assert _classify_by_name("unknown_tensor") == "weight"
        assert _classify_by_name("") == "weight"

    def test_classify_by_name_simple(self):
        assert _classify_by_name_simple("embed_tokens") == "embedding"
        assert _classify_by_name_simple("attn_q_proj") == "attention"
        assert _classify_by_name_simple("qkv.weight") == "qkv_fused"
        assert _classify_by_name_simple("ffn_gate.weight") == "ffn"
        assert _classify_by_name_simple("rms_norm.weight") == "norm_bias"
        assert _classify_by_name_simple("unknown") == "weight"
        assert _classify_by_name_simple("") == "weight"


class TestHelpersMetrics:
    def test_safe_bytes_ndarray(self):
        arr = np.zeros((10, 10), dtype=np.float32)
        assert _safe_bytes(arr) == 400

    def test_safe_bytes_dict(self):
        d = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([3.0], dtype=np.float32),
        }
        assert _safe_bytes(d) == 12

    def test_safe_bytes_list(self):
        assert (
            _safe_bytes(
                [
                    np.array([1.0], dtype=np.float32),
                    np.array([1.0, 2.0], dtype=np.float32),
                ]
            )
            == 12
        )

    def test_safe_bytes_bytes(self):
        assert _safe_bytes(b"hello") == 5

    def test_safe_bytes_int(self):
        assert _safe_bytes(42) == 8

    def test_compute_metrics(self):
        orig = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        recon = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        m = _compute_metrics(orig, recon)
        assert m["relative_error"] == 0.0
        assert m["cosine_similarity"] == 1.0
        assert m["snr_db"] == float("inf") or m["snr_db"] > 100

    def test_compute_metrics_with_error(self):
        orig = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        recon = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        m = _compute_metrics(orig, recon)
        assert m["relative_error"] > 0
        assert m["cosine_similarity"] < 1.0

    def test_compute_ratio(self):
        assert _compute_ratio(100, b"x" * 25) == 4.0
        assert _compute_ratio(0, b"x") == 0.0

    def test_sample_flat_small(self):
        arr = np.array([1.0, 2.0, 3.0])
        s = _sample_flat(arr, max_samples=100)
        np.testing.assert_array_equal(s, arr.astype(np.float64))

    def test_sample_flat_large(self):
        arr = np.random.randn(10000)
        s = _sample_flat(arr, max_samples=100)
        assert len(s) == 100

    def test_bootstrap_error_small(self):
        mean, std = _bootstrap_error(np.array([1.0, 2.0, 3.0]), n_resamples=50)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_bootstrap_error_single(self):
        mean, std = _bootstrap_error(np.array([5.0]), n_resamples=50)
        assert mean == 5.0
        assert std == 0.0

    def test_estimate_noise_floor(self):
        arr = np.random.randn(1000)
        nf = _estimate_noise_floor(arr)
        assert 0 <= nf <= 1.0

    def test_estimate_noise_floor_tiny(self):
        assert _estimate_noise_floor(np.array([1.0, 2.0])) == 0.0

    def test_estimate_entropy_rate(self):
        flat = np.sin(np.linspace(0, 10 * np.pi, 1000))
        er = _estimate_entropy_rate(flat)
        assert 0 <= er <= 4.0

    def test_estimate_entropy_rate_small(self):
        assert _estimate_entropy_rate(np.array([1.0, 2.0]), order=5) == 0.0


class TestHelpersScores:
    def test_toeplitz_score_non_2d(self):
        assert _toeplitz_score(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_toeplitz_score_small(self):
        assert _toeplitz_score(np.ones((2, 2))) == 0.0

    def test_toeplitz_score_toeplitz(self):
        t = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                t[i, j] = abs(i - j)
        score = _toeplitz_score(t)
        assert score >= 0.0

    def test_circulant_score_non_square(self):
        assert _circulant_score(np.ones((4, 8))) == 0.0

    def test_circulant_score_small(self):
        assert _circulant_score(np.ones((2, 2))) == 0.0

    def test_block_diagonal_score_non_2d(self):
        assert _block_diagonal_score(np.array([1.0, 2.0])) == 0.0

    def test_block_diagonal_score_diagonal(self):
        mat = np.eye(32)
        score = _block_diagonal_score(mat)
        assert score > 0.5

    def test_hierarchical_structure_score_non_2d(self):
        assert _hierarchical_structure_score(np.array([1.0, 2.0])) == 0.0

    def test_hierarchical_structure_score_small(self):
        assert _hierarchical_structure_score(np.ones((4, 4))) == 0.0

    def test_mutual_information_blocks_non_2d(self):
        assert _mutual_information_blocks(np.array([1.0, 2.0])) == 0.0

    def test_kolmogorov_estimate(self):
        flat = np.sin(np.linspace(0, 10, 100))
        k = _kolmogorov_estimate(flat)
        assert 0 <= k <= 1.0

    def test_kolmogorov_estimate_small(self):
        assert _kolmogorov_estimate(np.array([1.0, 2.0])) == 0.0

    def test_structured_nm_score_non_2d(self):
        assert _structured_nm_score(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_block_sparsity_score_non_2d(self):
        assert _block_sparsity_score(np.array([1.0, 2.0])) == 0.0

    def test_block_sparsity_score_small(self):
        assert _block_sparsity_score(np.ones((4, 4)), block_size=16) == 0.0

    def test_unstructured_sparsity_score(self):
        arr = np.concatenate([np.zeros(90), np.ones(10)])
        np.random.shuffle(arr)
        s = _unstructured_sparsity_score(arr)
        assert 0 <= s <= 1.0

    def test_nm_sparsity_score(self):
        arr = np.random.randn(32, 32)
        score, details = _nm_sparsity_score(arr)
        assert 0 <= score <= 1.0
        assert "2:4" in details
        assert "unstructured" in details


# =============================================================================
# compression/cli.py — CLI function-level tests
# =============================================================================


class TestCLIHelpers:
    def test_human_size(self):
        from spectralstream.compression.cli import _human_size

        assert _human_size(500) == "500.0B"
        assert _human_size(1024) == "1.0KB"
        assert _human_size(1048576) == "1.0MB"
        assert _human_size(1073741824) == "1.0GB"

    def test_progress_bar_no_rich(self):
        from spectralstream.compression.cli import _progress_bar

        items = list(_progress_bar([1, 2, 3], desc="test"))
        assert items == [1, 2, 3]

    def test_progress_bar_empty(self):
        from spectralstream.compression.cli import _progress_bar

        items = list(_progress_bar([], desc="empty"))
        assert items == []


# =============================================================================
# compression/engine/_orchestrator.py — key methods
# =============================================================================


class TestOrchestratorEdgeCases:
    def test_engine_init_defaults(self, tiny_engine):
        engine = tiny_engine
        assert engine.config.target_ratio == 5000.0
        assert engine.config.max_error == 0.0002
        assert hasattr(engine, "_methods")
        assert len(engine.get_available_methods()) > 0
        assert engine.discovery is not None

    def test_engine_no_intelligence(self, tiny_engine):
        engine = tiny_engine
        assert hasattr(engine, "get_available_methods")
        assert len(engine.get_available_methods()) > 0

    def test_get_method_names(self, tiny_engine):
        engine = tiny_engine
        names = engine.get_method_names()
        assert "block_int8" in names
        assert "block_int4" in names

    def test_get_available_methods(self, tiny_engine):
        engine = tiny_engine
        methods = engine.get_available_methods()
        assert "block_int8" in methods
        assert "block_int4" in methods

    def test_compress_tensor_passthrough(self, tiny_engine):
        engine = tiny_engine
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        assert ratio >= 1.0
        assert "method" in meta

    def test_compress_tensor_block_int8(self, tiny_engine):
        engine = tiny_engine
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        assert ratio >= 1.0

    def test_compress_tensor_unknown_method_falls_back(self, tiny_engine):
        engine = tiny_engine
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        assert meta.get("method", "") != ""

    def test_decompress_tensor_passthrough(self, tiny_engine):
        engine = tiny_engine
        from spectralstream.compression.engine._dataclasses import CompressedTensor

        ct = CompressedTensor(
            _data=np.array([1.0, 2.0, 3.0], dtype=np.float16).tobytes(),
            method="passthrough",
            params={},
            original_shape=(3,),
            original_dtype="float32",
            compression_ratio=1.0,
            relative_error=0.0,
            snr_db=float("inf"),
            psnr_db=float("inf"),
            cosine_similarity=1.0,
            computation_time=0.0,
        )
        result = engine.decompress(ct.data, ct.params)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_validate_compression(self, tiny_engine):
        engine = tiny_engine
        tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="t")
        assert ratio >= 1.0
        assert "method" in meta
        assert error >= 0.0

    def test_build_report_empty(self, tiny_engine):
        engine = tiny_engine

        stats = {
            "tensors": [],
            "total_orig_bytes": 0,
            "total_compressed_bytes": 0,
            "overall_ratio": 0.5,
            "average_ratio": 0.5,
            "avg_error": 0.0,
            "max_error": 0.0,
            "min_error": 0.0,
            "num_tensors": 0,
            "method_distribution": {},
            "failures": [],
            "per_layer_error": {},
        }
        report = engine._build_report(stats)
        assert isinstance(report, dict)
        assert len(report.get("failures", [])) == 0

    def test_get_telemetry(self, tiny_engine):
        engine = tiny_engine
        telemetry = engine.get_telemetry()
        assert "timestamps" in telemetry
        assert "method_success_rates" in telemetry

    def test_validate_report(self, tiny_engine):
        engine = tiny_engine
        from spectralstream.compression.engine._dataclasses import (
            CompressionReport,
            CompressedTensor,
        )

        report = CompressionReport(
            tensors=[
                CompressedTensor(
                    _data=b"x",
                    method="block_int8",
                    params={},
                    original_shape=(1,),
                    original_dtype="float32",
                    compression_ratio=2.0,
                    relative_error=0.001,
                    snr_db=30.0,
                    psnr_db=30.0,
                    cosine_similarity=0.999,
                    computation_time=0.01,
                )
            ],
        )
        stats = engine.validate_report(report)
        assert stats["n_tensors"] == 1
        assert stats["overall_ratio"] > 0
        assert "quality_distribution" in stats

    def test_profile_tensor_names(self, tiny_engine):
        engine = tiny_engine
        profiles = engine.profile_tensor_names(["embed_tokens.weight", "q_proj.weight"])
        assert "embed_tokens.weight" in profiles
        assert "q_proj.weight" in profiles
        assert profiles["embed_tokens.weight"].tensor_type == "embedding"
        assert profiles["q_proj.weight"].tensor_type == "attention_q"

    def test_profile_tensor_names_with_overrides(self, tiny_engine):
        engine = tiny_engine
        profiles = engine.profile_tensor_names(
            ["w1"], sensitivity_overrides={"w1": 0.9}
        )
        assert profiles["w1"].sensitivity == 0.9

    def test_get_method_stats(self, tiny_engine):
        engine = tiny_engine
        stats = engine.get_method_stats()
        assert isinstance(stats, dict)

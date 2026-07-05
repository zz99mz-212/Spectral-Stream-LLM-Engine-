import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.format.core import TensorDType
    from spectralstream.format.index import (
        TensorIndexEntry,
        TensorIndex,
        LegacyTensorIndexEntry,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestTensorIndexEntry:
    def test_create_entry(self):
        entry = TensorIndexEntry(
            name="test.tensor",
            shape=(64, 64),
            dtype=TensorDType.F32,
            compression_method=1,
            compression_params={"bits": 8},
            data_offset=1024,
            compressed_size=4096,
            original_size=16384,
            quality_metrics={"snr_db": 30.0, "compression_ratio": 4.0},
            checksum=b"abcdefghijklmnopqrstuvwxyz123456",
        )
        assert entry.name == "test.tensor"
        assert entry.shape == (64, 64)
        assert entry.dtype == TensorDType.F32

    def test_pack_unpack_roundtrip(self):
        entry = TensorIndexEntry(
            name="pack.test",
            shape=(128,),
            dtype=TensorDType.F16,
            compression_method=2,
            compression_params={"method": "fwht"},
            data_offset=2048,
            compressed_size=512,
            original_size=1024,
            quality_metrics={"snr_db": 40.0, "compression_ratio": 2.0},
            checksum=b"a" * 32,
            flags=1,
        )
        packed = entry.pack()
        unpacked, size = TensorIndexEntry.unpack(packed)
        assert unpacked.name == entry.name
        assert unpacked.shape == entry.shape
        assert unpacked.dtype == entry.dtype
        assert unpacked.compression_method == entry.compression_method
        assert unpacked.data_offset == entry.data_offset
        assert unpacked.compressed_size == entry.compressed_size
        assert unpacked.flags == entry.flags

    def test_unpack_empty_raises(self):
        import pytest

        with pytest.raises(Exception):
            TensorIndexEntry.unpack(b"")

    def test_pack_empty_params(self):
        entry = TensorIndexEntry(
            name="no_params",
            shape=(10,),
            dtype=TensorDType.U8,
            compression_method=0,
            compression_params={},
            data_offset=0,
            compressed_size=0,
            original_size=40,
            quality_metrics={},
            checksum=b"\x00" * 32,
        )
        packed = entry.pack()
        unpacked, _ = TensorIndexEntry.unpack(packed)
        assert unpacked.name == "no_params"


class TestTensorIndex:
    def test_add_and_get(self):
        idx = TensorIndex()
        entry = TensorIndexEntry(
            name="test",
            shape=(4,),
            dtype=TensorDType.F32,
            compression_method=0,
            compression_params={},
            data_offset=0,
            compressed_size=0,
            original_size=16,
            quality_metrics={},
            checksum=b"\x00" * 32,
        )
        idx.add(entry)
        assert idx.get("test") is entry
        assert idx[0] is entry
        assert len(idx) == 1

    def test_names(self):
        idx = TensorIndex()
        for n in ["a", "b", "c"]:
            entry = TensorIndexEntry(
                name=n,
                shape=(1,),
                dtype=TensorDType.F32,
                compression_method=0,
                compression_params={},
                data_offset=0,
                compressed_size=0,
                original_size=4,
                quality_metrics={},
                checksum=b"\x00" * 32,
            )
            idx.add(entry)
        assert idx.names() == ["a", "b", "c"]

    def test_iteration(self):
        idx = TensorIndex()
        for n in ["x", "y"]:
            entry = TensorIndexEntry(
                name=n,
                shape=(1,),
                dtype=TensorDType.F32,
                compression_method=0,
                compression_params={},
                data_offset=0,
                compressed_size=0,
                original_size=4,
                quality_metrics={},
                checksum=b"\x00" * 32,
            )
            idx.add(entry)
        names = [e.name for e in idx]
        assert names == ["x", "y"]

    def test_pack_unpack_index(self):
        idx = TensorIndex()
        for n in ["t1", "t2"]:
            entry = TensorIndexEntry(
                name=n,
                shape=(2, 2),
                dtype=TensorDType.F32,
                compression_method=1,
                compression_params={"k": "v"},
                data_offset=100,
                compressed_size=32,
                original_size=64,
                quality_metrics={"snr_db": 20.0},
                checksum=b"b" * 32,
            )
            idx.add(entry)
        packed = idx.pack()
        idx2 = TensorIndex.unpack(packed)
        assert len(idx2) == 2
        assert idx2.get("t1") is not None
        assert idx2.get("t2") is not None

    def test_legacy_unpack(self):
        import struct

        name = b"legacy_tensor"
        buf = struct.pack("<I", len(name)) + name
        buf += struct.pack("<I", 2)
        buf += struct.pack("<QQ", 3, 4)
        buf += struct.pack("<HH", 0, 2)
        buf += struct.pack("<QQQ", 500, 200, 600)
        buf += b"\x00" * 32
        buf += struct.pack("<I", 0)
        idx = TensorIndex.unpack(buf, is_legacy=True)
        assert len(idx) == 1
        entry = idx[0]
        assert entry.name == "legacy_tensor"
        assert entry.shape == (3, 4)

    def test_empty_index(self):
        idx = TensorIndex()
        assert len(idx) == 0
        assert idx.names() == []


class TestLegacyTensorIndexEntry:
    def test_unpack_legacy(self):
        import struct

        name = b"old_tensor"
        buf = struct.pack("<I", len(name)) + name
        buf += struct.pack("<I", 1)
        buf += struct.pack("<Q", 128)
        buf += struct.pack("<HH", 1, 3)
        buf += struct.pack("<QQQ", 0, 100, 512)
        buf += b"\x00" * 32
        buf += struct.pack("<I", 0)
        entry, size = LegacyTensorIndexEntry.unpack(buf)
        assert entry.name == "old_tensor"
        assert entry.shape == (128,)
        assert entry.compression == 3

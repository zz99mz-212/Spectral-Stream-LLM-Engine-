import sys

sys.path.insert(0, ".")
try:
    from spectralstream.format.core import SSF_MAGIC, SSF_REDUNDANT_HEADER_OFFSET
    from spectralstream.format.header import SSFHeader
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestSSFHeader:
    def test_default_header(self):
        hdr = SSFHeader()
        assert hdr.magic == SSF_MAGIC
        assert hdr.version == 3
        assert hdr.flags == 0
        assert hdr.n_tensors == 0
        assert hdr.redundant_header_offset == SSF_REDUNDANT_HEADER_OFFSET

    def test_custom_header(self):
        hdr = SSFHeader(
            version=2,
            flags=1,
            n_tensors=10,
            index_offset=1024,
            index_size=2048,
            metadata_offset=4096,
            metadata_size=512,
            tensor_data_offset=8192,
        )
        assert hdr.version == 2
        assert hdr.flags == 1
        assert hdr.n_tensors == 10
        assert hdr.index_offset == 1024
        assert hdr.index_size == 2048

    def test_pack_returns_256_bytes(self):
        hdr = SSFHeader()
        packed = hdr.pack()
        assert len(packed) == 256

    def test_pack_unpack_roundtrip(self):
        original = SSFHeader(
            version=3,
            flags=1,
            n_tensors=5,
            index_offset=2048,
            index_size=1024,
            metadata_offset=4096,
            metadata_size=256,
            tensor_data_offset=65536,
            footer_offset=131072,
        )
        packed = original.pack()
        unpacked = SSFHeader.unpack(packed)
        assert unpacked.magic == original.magic
        assert unpacked.version == original.version
        assert unpacked.flags == original.flags
        assert unpacked.n_tensors == original.n_tensors
        assert unpacked.index_offset == original.index_offset
        assert unpacked.index_size == original.index_size
        assert unpacked.footer_offset == original.footer_offset

    def test_unpack_bad_magic(self):
        import pytest

        bad_data = b"\x00" * 256
        with pytest.raises(ValueError, match="Bad magic"):
            SSFHeader.unpack(bad_data)

    def test_unpack_bad_version(self):
        import pytest
        import struct
        from spectralstream.format.core import SSF_HEADER_SIZE

        hdr = SSFHeader(version=99)
        packed = hdr.pack()
        with pytest.raises(ValueError, match="Unsupported SSF version"):
            packed = packed[:4] + struct.pack("<I", 99) + packed[8:]
            SSFHeader.unpack(packed)

    def test_unpack_too_small(self):
        import pytest

        with pytest.raises(ValueError, match="Header too small"):
            SSFHeader.unpack(b"\x00" * 100)

    def test_roundtrip_zero_tensors(self):
        hdr = SSFHeader(n_tensors=0)
        packed = hdr.pack()
        unpacked = SSFHeader.unpack(packed)
        assert unpacked.n_tensors == 0

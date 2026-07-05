import pytest
import sys

sys.path.insert(0, ".")
try:
    import struct
    import numpy as np
    from spectralstream.format.gguf_parser_engine import (
        GGUFParser,
        GGMLDequantizer,
        GGUF_MAGIC,
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q8_0,
        GGML_TYPE_BF16,
        GGML_BLOCK_SIZE,
        GGML_BLOCK_BYTES,
        GGML_TYPE_NAMES,
        _bf16_to_f32,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


def _make_minimal_gguf(tensor_count=0):
    buf = GGUF_MAGIC
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", tensor_count)
    buf += struct.pack("<Q", 0)
    buf = bytearray(buf)
    padding = (32 - (len(buf) % 32)) % 32
    buf.extend(b"\x00" * padding)
    return bytes(buf)


class TestGGUFConstants:
    def test_magic(self):
        assert GGUF_MAGIC == b"GGUF"

    def test_block_sizes(self):
        assert GGML_BLOCK_SIZE[GGML_TYPE_F32] == 1
        assert GGML_BLOCK_SIZE[GGML_TYPE_Q4_0] == 32
        assert GGML_BLOCK_SIZE[GGML_TYPE_Q8_0] == 32

    def test_block_bytes(self):
        assert GGML_BLOCK_BYTES[GGML_TYPE_F32] == 4
        assert GGML_BLOCK_BYTES[GGML_TYPE_F16] == 2
        assert GGML_BLOCK_BYTES[GGML_TYPE_Q4_0] == 18

    def test_type_names(self):
        assert GGML_TYPE_NAMES[GGML_TYPE_F32] == "F32"
        assert GGML_TYPE_NAMES[GGML_TYPE_Q4_0] == "Q4_0"
        assert GGML_TYPE_NAMES[GGML_TYPE_Q8_0] == "Q8_0"


class TestBf16ToF32:
    def test_bf16_conversion(self):
        result = _bf16_to_f32(0x3F80)
        assert abs(result - 1.0) < 0.01

    def test_bf16_zero(self):
        result = _bf16_to_f32(0)
        assert result == 0.0


class TestGGUFParser:
    def test_from_buffer_basic(self):
        data = _make_minimal_gguf(tensor_count=0)
        parser = GGUFParser.from_buffer(data)
        assert parser.magic == GGUF_MAGIC
        assert parser.version == 3
        assert parser.tensor_count == 0

    def test_parse_empty_metadata(self):
        data = _make_minimal_gguf(tensor_count=0)
        parser = GGUFParser.from_buffer(data)
        assert len(parser.metadata) == 0
        assert len(parser.tensor_infos) == 0

    def test_parser_repr(self):
        data = _make_minimal_gguf(tensor_count=0)
        parser = GGUFParser.from_buffer(data)
        r = repr(parser)
        assert "GGUFParser" in r

    def test_summary(self):
        data = _make_minimal_gguf(tensor_count=0)
        parser = GGUFParser.from_buffer(data)
        s = parser.summary()
        assert "Version: 3" in s

    def test_bad_magic(self):
        with pytest.raises(ValueError, match="Not a GGUF file"):
            GGUFParser.from_buffer(b"NOTGGUF")

    def test_get_tensor_info_nonexistent(self):
        data = _make_minimal_gguf(tensor_count=0)
        parser = GGUFParser.from_buffer(data)
        assert parser.get_tensor_info("nonexistent") is None


class TestGGMLDequantizer:
    def test_f32_passthrough(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        data_arr = np.frombuffer(data, dtype=np.uint8)
        result = GGMLDequantizer.dequantize_fast(data_arr, GGML_TYPE_F32)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_f16_dequant(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16).tobytes()
        data_arr = np.frombuffer(data, dtype=np.uint8)
        result = GGMLDequantizer.dequantize_fast(data_arr, GGML_TYPE_F16)
        assert np.allclose(result, [1.0, 2.0, 3.0], atol=0.01)

    def test_q4_0_dequant_shape(self):
        d = struct.pack("<e", 1.0)
        nibbles = bytes(
            [
                0x10,
                0x32,
                0x54,
                0x76,
                0x98,
                0xBA,
                0xDC,
                0xFE,
                0x01,
                0x23,
                0x45,
                0x67,
                0x89,
                0xAB,
                0xCD,
                0xEF,
            ]
        )
        data_block = d + nibbles
        assert len(data_block) == 18
        data_arr = np.frombuffer(data_block, dtype=np.uint8)
        result = GGMLDequantizer.dequantize_fast(data_arr, GGML_TYPE_Q4_0)
        assert result.size == 32

    def test_bf16_dequant(self):
        raw = struct.pack("<H", 0x3F80)
        data_arr = np.frombuffer(raw, dtype=np.uint8)
        result = GGMLDequantizer.dequantize_fast(data_arr, GGML_TYPE_BF16)
        assert abs(result[0] - 1.0) < 0.01

    def test_unknown_type_fallback(self):
        data = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        data_arr = np.frombuffer(data, dtype=np.uint8)
        result = GGMLDequantizer.dequantize_fast(data_arr, 9999)
        assert result.size > 0

    def test_q8_0_fast_shape(self):
        d = struct.pack("<e", 0.5)
        qs = struct.pack(
            "<32b",
            *[
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                -1,
                -2,
                -3,
                -4,
                -5,
                -6,
                -7,
                -8,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )
        data_block = d + qs
        assert len(data_block) == 34
        data_arr = np.frombuffer(data_block, dtype=np.uint8)
        result = GGMLDequantizer.dequantize_fast(data_arr, GGML_TYPE_Q8_0)
        assert result.size == 32

    def test_deq_q8_0_legacy(self):
        d = struct.pack("<e", 0.5)
        qs = struct.pack("<32b", *([1] * 32))
        data_block = d + qs
        data_arr = np.frombuffer(data_block, dtype=np.uint8)
        result = GGMLDequantizer.dequantize(data_arr, GGML_TYPE_Q8_0)
        assert result.size == 32

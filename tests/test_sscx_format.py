"""Tests for SSCX format and ModelCompressor."""

import json
import os
import struct
import tempfile
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)

from spectralstream.sscx_format import (
    SSCXWriter,
    SSCXReader,
    SSCXHeader,
    SSCXTensorInfo,
    SSCXLayerInfo,
    SSCXFooter,
    COMP_RAW,
    COMP_DCT,
    COMP_INT8,
    COMP_INT4,
    COMP_DELTA,
    COMP_NAMES,
    DTYPE_FP32,
    DTYPE_FP16,
    _align_up,
    _format_size,
    _crc32,
    page_align,
    page_aligned_size,
)
from spectralstream.model_compressor import (
    ModelCompressor,
    TensorProfile,
    CompressionReport,
    ValidationReport,
    _compress_int8,
    _decompress_int8,
    _compress_int4,
    _decompress_int4,
    _compress_dct_block,
    _decompress_dct_block,
    _error_metrics,
    _get_sensitivity,
    _detect_layer_id,
    _is_embedding,
    _zigzag_scan,
    _zigzag_unscan,
)

PASS: int = 0
FAIL: int = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Assert a condition, incrementing the global PASS/FAIL counters.

    Parameters
    ----------
    label : str
        Human-readable test label printed in the result message.
    condition : bool
        The condition to evaluate.
    detail : str, optional
        Additional detail appended on failure.
    """
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def test_utilities() -> None:
    """Test alignment, formatting, page-size, and CRC32 utility functions."""
    print("\n=== Utility Functions ===")
    check("_align_up(100, 64) == 128", _align_up(100, 64) == 128)
    check("_align_up(128, 64) == 128", _align_up(128, 64) == 128)
    check("_align_up(0, 4096) == 0", _align_up(0, 4096) == 0)
    check("_format_size(1024) == '1.0 KB'", _format_size(1024) == "1.0 KB")
    check("_format_size(1048576) == '1.0 MB'", _format_size(1048576) == "1.0 MB")
    check(
        "_format_size(1073741824) == '1.00 GB'", _format_size(1073741824) == "1.00 GB"
    )
    check("page_aligned_size(100) == 4096", page_aligned_size(100) == 4096)
    check("page_aligned_size(4096) == 4096", page_aligned_size(4096) == 4096)
    data = b"hello world"
    check("CRC32 is deterministic", _crc32(data) == _crc32(data))
    check("CRC32 is 32-bit", 0 <= _crc32(data) <= 0xFFFFFFFF)


def test_header_pack_unpack() -> None:
    """Test SSCXHeader serialisation and round-trip deserialisation."""
    print("\n=== Header Pack/Unpack ===")
    h = SSCXHeader(
        version=1,
        flags=3,
        num_tensors=100,
        num_layers=12,
        total_original_bytes=1024 * 1024,
        total_compressed_bytes=2048,
        target_ratio=500.0,
        max_error=0.0002,
        model_name="test-model",
    )
    packed = h.pack()
    check("Header has correct size", len(packed) >= 108)
    check("Header starts with SSCX", packed[:4] == b"SSCX")

    h2 = SSCXHeader.unpack(packed)
    check("Round-trip version", h2.version == 1)
    check("Round-trip flags", h2.flags == 3)
    check("Round-trip num_tensors", h2.num_tensors == 100)
    check("Round-trip num_layers", h2.num_layers == 12)
    check("Round-trip total_original", h2.total_original_bytes == 1024 * 1024)
    check("Round-trip total_compressed", h2.total_compressed_bytes == 2048)
    check("Round-trip target_ratio", h2.target_ratio == 500.0)
    check("Round-trip max_error", abs(h2.max_error - 0.0002) < 0.0001)
    check("Round-trip model_name", h2.model_name == "test-model")


def test_header_bad_magic() -> None:
    """Test that an invalid magic number raises ``ValueError``."""
    print("\n=== Header Bad Magic ===")
    bad = b"BAD\x00" + b"\x00" * 124
    try:
        SSCXHeader.unpack(bad)
        check("Should raise ValueError", False)
    except ValueError as e:
        check("Raises on bad magic", "Bad SSCX magic" in str(e))


def test_tensor_info_pack_unpack() -> None:
    """Test SSCXTensorInfo serialisation and round-trip."""
    print("\n=== TensorInfo Pack/Unpack ===")
    t = SSCXTensorInfo(
        name="blk.0.attn.q.weight",
        offset=4096,
        compressed_size=1024,
        original_size=4096,
        shape=(64, 64),
        dtype_code=DTYPE_FP32,
        compression_method=COMP_INT8,
        error_snr=45.0,
        error_rel=0.001,
        error_psnr=50.0,
        error_cos=0.9999,
        block_checksum=0xDEADBEEF,
        layer_id=0,
    )
    packed = t.pack()
    t2 = SSCXTensorInfo.unpack(packed)
    check("Round-trip name", t2.name == "blk.0.attn.q.weight")
    check("Round-trip offset", t2.offset == 4096)
    check("Round-trip compressed_size", t2.compressed_size == 1024)
    check("Round-trip original_size", t2.original_size == 4096)
    check("Round-trip shape", t2.shape == (64, 64))
    check("Round-trip dtype", t2.dtype_code == DTYPE_FP32)
    check("Round-trip compression", t2.compression_method == COMP_INT8)
    check("Round-trip error_snr", abs(t2.error_snr - 45.0) < 0.01)
    check("Round-trip block_checksum", t2.block_checksum == 0xDEADBEEF)
    check("Round-trip layer_id", t2.layer_id == 0)


def test_layer_info_pack_unpack() -> None:
    """Test SSCXLayerInfo serialisation and round-trip."""
    print("\n=== LayerInfo Pack/Unpack ===")
    l = SSCXLayerInfo(
        layer_id=5,
        num_tensors=8,
        cross_layer_group=1,
        compression_params=[1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    packed = l.pack()
    check(
        "Layer index is 40 bytes", len(packed) == SSCXLayerInfo.LAYER_INDEX_ENTRY_SIZE
    )
    l2 = SSCXLayerInfo.unpack(packed)
    check("Round-trip layer_id", l2.layer_id == 5)
    check("Round-trip num_tensors", l2.num_tensors == 8)
    check("Round-trip cross_layer_group", l2.cross_layer_group == 1)
    check("Round-trip params[0]", abs(l2.compression_params[0] - 1.0) < 0.001)
    check("Round-trip params[2]", abs(l2.compression_params[2] - 0.5) < 0.001)


def test_footer_pack_unpack() -> None:
    """Test SSCXFooter serialisation and round-trip."""
    print("\n=== Footer Pack/Unpack ===")
    f = SSCXFooter(crc32=0xCAFEBABE, tensor_index_offset=512, layer_index_offset=128)
    packed = f.pack()
    check("Footer is 12 bytes", len(packed) == 12)
    f2 = SSCXFooter.unpack(packed)
    check("Round-trip crc32", f2.crc32 == 0xCAFEBABE)
    check("Round-trip tensor_index_offset", f2.tensor_index_offset == 512)
    check("Round-trip layer_index_offset", f2.layer_index_offset == 128)


def test_write_read_roundtrip() -> None:
    """Test full SSCX write-then-read round-trip with synthetic tensors."""
    print("\n=== Write/Read Round-trip ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.sscx")

        writer = SSCXWriter(path, model_name="test-model", target_ratio=100.0)

        rng = np.random.RandomState(42)
        tensors = {
            "embed.weight": rng.randn(100, 32).astype(np.float32),
            "blk.0.q_proj.weight": rng.randn(32, 32).astype(np.float32),
            "blk.0.k_proj.weight": rng.randn(32, 32).astype(np.float32),
            "blk.1.q_proj.weight": rng.randn(32, 32).astype(np.float32),
            "lm_head.weight": rng.randn(100, 32).astype(np.float32),
        }

        for name, tensor in tensors.items():
            writer.add_tensor(name, tensor, layer_id=_detect_layer_id(name))

        stats = writer.save()
        check("File exists", os.path.exists(path))
        check("Total tensors = 5", stats["num_tensors"] == 5)
        check("Ratio >= 1", stats["ratio"] >= 1.0)

        reader = SSCXReader(path)
        check("Reader header version", reader.header.version == 1)
        check("Reader model_name", reader.header.model_name == "test-model")
        check("Reader list_tensors", len(reader.list_tensors()) == 5)

        for name, original in tensors.items():
            loaded = reader.get_tensor(name)
            check(f"Read {name} shape match", loaded.shape == original.shape)
            check(f"Read {name} values match", np.allclose(loaded, original, atol=1e-6))

        summary = reader.summary()
        check("Summary has path", "path" in summary)
        check("Summary has num_tensors", summary["num_tensors"] == 5)
        reader.close()


def test_write_read_precompressed() -> None:
    """Test writing and reading pre-compressed INT8 tensor blocks."""
    print("\n=== Write/Read Pre-compressed Blocks ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_comp.sscx")
        writer = SSCXWriter(path, model_name="compressed-test")

        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 64).astype(np.float32)

        # Compress to INT8
        block, comp_size = _compress_int8(tensor)
        recon_raw = _decompress_int8(block, tensor.size)
        recon = recon_raw.reshape(tensor.shape)
        metrics = _error_metrics(tensor, recon)

        writer.add_tensor_block(
            "weight",
            block,
            tensor.nbytes,
            tensor.shape,
            DTYPE_FP32,
            0,
            COMP_INT8,
            metrics,
        )
        stats = writer.save()
        check("Pre-compressed write", os.path.exists(path))

        reader = SSCXReader(path)
        loaded = reader.get_tensor("weight")
        check("Pre-compressed read shape", loaded.shape == tensor.shape)

        # Verify decompression round-trip
        np.testing.assert_allclose(loaded, recon, atol=1e-5)
        check("Pre-compressed values match", True)
        reader.close()


def test_int8_compress_decompress() -> None:
    """Test INT8 compression and decompression round-trip fidelity."""
    print("\n=== INT8 Compress/Decompress ===")
    rng = np.random.RandomState(42)
    tensor = rng.randn(256).astype(np.float32)
    block, comp_size = _compress_int8(tensor)
    check("INT8 comp_size < original", len(block) < tensor.nbytes)

    recon_raw = _decompress_int8(block, tensor.size)
    recon = recon_raw.reshape(tensor.shape)
    metrics = _error_metrics(tensor, recon)
    check(f"INT8 SNR > 30 dB", metrics["snr"] > 30.0)
    check(f"INT8 relative error < 5%", metrics["rel"] < 0.05)

    # Test with 2D tensor
    mat = rng.randn(32, 32).astype(np.float32)
    block2, _ = _compress_int8(mat)
    recon2 = _decompress_int8(block2, mat.size).reshape(mat.shape)
    check("INT8 2D shape preserved", recon2.shape == mat.shape)
    m2 = _error_metrics(mat, recon2)
    check("INT8 2D SNR > 30 dB", m2["snr"] > 30.0)


def test_int4_compress_decompress() -> None:
    """Test INT4 compression and decompression round-trip fidelity."""
    print("\n=== INT4 Compress/Decompress ===")
    rng = np.random.RandomState(42)
    tensor = rng.randn(128).astype(np.float32)
    block, comp_size = _compress_int4(tensor)
    check("INT4 comp_size < INT8", len(block) < tensor.nbytes // 2)

    recon_raw = _decompress_int4(block, tensor.size)
    recon = recon_raw.reshape(tensor.shape)
    metrics = _error_metrics(tensor, recon)
    check(f"INT4 SNR > 10 dB", metrics["snr"] > 10.0)
    check(f"INT4 relative error < 25%", metrics["rel"] < 0.25)


def test_dct_compress_decompress() -> None:
    """Test DCT block compression and decompression round-trip."""
    print("\n=== DCT Block Compress/Decompress ===")
    rng = np.random.RandomState(42)
    mat = rng.randn(64, 64).astype(np.float32)
    block, comp_size = _compress_dct_block(mat)
    check("DCT comp_size < original", len(block) < mat.nbytes)

    recon = _decompress_dct_block(block, mat.shape)
    check("DCT shape preserved", recon.shape == mat.shape)
    metrics = _error_metrics(mat, recon)
    check(f"DCT SNR > 0 dB (random data)", metrics["snr"] > 0.0)
    check(f"DCT cosine > 0.85", metrics["cos"] > 0.85)


def test_zigzag() -> None:
    """Test zigzag scan and its inverse (unscan) round-trip."""
    print("\n=== Zigzag Scan ===")
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    flat = _zigzag_scan(m)
    check("Zigzag output is 1D", flat.ndim == 1)
    check("Zigzag output has all elements", len(flat) == 9)
    check("Zigzag first element is 1", flat[0] == 1.0)

    reconstructed = _zigzag_unscan(flat, 3)
    check("Zigzag round-trip", np.allclose(m, reconstructed))


def test_error_metrics() -> None:
    """Test ``_error_metrics`` with exact match and noisy reconstructions."""
    print("\n=== Error Metrics ===")
    orig = np.array([1.0, 2.0, 3.0, 4.0])
    recon = np.array([1.0, 2.0, 3.0, 4.0])
    m = _error_metrics(orig, recon)
    check("Perfect match SNR=100", m["snr"] >= 99.0)
    check("Perfect match rel=0", m["rel"] < 1e-10)
    check("Perfect match cos=1", abs(m["cos"] - 1.0) < 1e-10)

    noisy = orig + np.array([0.01, -0.01, 0.02, -0.01])
    m2 = _error_metrics(orig, noisy)
    check("Noisy SNR > 30", m2["snr"] > 30.0)
    check("Noisy cos > 0.999", m2["cos"] > 0.999)


def test_sensitivity() -> None:
    """Test sensitivity scoring and embedding detection helpers."""
    print("\n=== Sensitivity Detection ===")
    check("embed high sensitivity", _get_sensitivity("embed.weight") > 0.8)
    check("lm_head highest sensitivity", _get_sensitivity("lm_head.weight") >= 0.95)
    check("bias lower sensitivity", _get_sensitivity("attn.bias") < 0.7)
    check("weight default sensitivity", _get_sensitivity("some.weight") == 0.5)
    check("is_embedding detects embed", _is_embedding("embed_tokens.weight"))
    check("is_embedding detects wte", _is_embedding("wte.weight"))
    check("is_embedding rejects bias", not _is_embedding("attn.bias"))


def test_layer_detection() -> None:
    """Test layer-ID extraction from tensor names."""
    print("\n=== Layer ID Detection ===")
    check("blk.0. -> 0", _detect_layer_id("blk.0.q_proj.weight") == 0)
    check("blk.15. -> 15", _detect_layer_id("blk.15.attn.weight") == 15)
    check("block.5. -> 5", _detect_layer_id("block.5.mlp.weight") == 5)
    check("layer.32. -> 32", _detect_layer_id("layer.32.norm.weight") == 32)
    check("no layer -> -1", _detect_layer_id("embed.weight") == -1)


def test_crc32() -> None:
    """Test CRC32 checksum determinism, range, and sensitivity to input."""
    print("\n=== CRC32 Integrity ===")
    data = b"test data for checksum"
    cs = _crc32(data)
    check("CRC32 is int", isinstance(cs, int))
    check("CRC32 in range", 0 <= cs <= 0xFFFFFFFF)
    check("CRC32 differs for different data", _crc32(data) != _crc32(data + b"!"))


def test_synthetic_model_compress() -> None:
    """Test end-to-end compress/decompress/validate on a synthetic model."""
    print("\n=== Synthetic Model Compression ===")
    rng = np.random.RandomState(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        sf_path = os.path.join(tmpdir, "model.safetensors")
        sscx_path = os.path.join(tmpdir, "model.sscx")

        # Create synthetic safetensors-like file
        model_tensors = {}
        shapes = {
            "embed_tokens.weight": (100, 32),
            "blk.0.self_attn.q_proj.weight": (32, 32),
            "blk.0.self_attn.k_proj.weight": (32, 32),
            "blk.0.self_attn.v_proj.weight": (32, 32),
            "blk.0.self_attn.o_proj.weight": (32, 32),
            "blk.0.mlp.gate_proj.weight": (64, 32),
            "blk.0.mlp.up_proj.weight": (64, 32),
            "blk.0.mlp.down_proj.weight": (32, 64),
            "blk.1.self_attn.q_proj.weight": (32, 32),
            "blk.1.self_attn.k_proj.weight": (32, 32),
            "blk.1.self_attn.v_proj.weight": (32, 32),
            "blk.1.self_attn.o_proj.weight": (32, 32),
            "blk.1.mlp.gate_proj.weight": (64, 32),
            "blk.1.mlp.up_proj.weight": (64, 32),
            "blk.1.mlp.down_proj.weight": (32, 64),
            "lm_head.weight": (100, 32),
        }

        # Write safetensors format
        header = {}
        offset = 0
        for name, shape in shapes.items():
            n = 1
            for d in shape:
                n *= d
            nbytes = n * 4
            header[name] = {
                "dtype": "F32",
                "data_offsets": [offset, offset + nbytes],
                "shape": list(shape),
            }
            offset += nbytes

        header_bytes = json.dumps(header).encode("utf-8")
        with open(sf_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for name, shape in shapes.items():
                n = 1
                for d in shape:
                    n *= d
                t = rng.randn(n).astype(np.float32)
                model_tensors[name] = t.reshape(shape)
                f.write(t.tobytes())

        # Compress
        compressor = ModelCompressor()
        report = compressor.compress(
            sf_path,
            sscx_path,
            target_ratio=100.0,
            max_error=0.01,
            quality="balanced",
        )
        print(report.summary())
        check("Compression produced report", report.total_tensors > 0)
        check("Compression ratio > 1", report.overall_ratio > 1.0)

        # Decompress
        decompressed = compressor.decompress(sscx_path)
        check("Decompressed all tensors", len(decompressed) == len(shapes))

        # Validate
        validation = compressor.validate(sf_path, sscx_path, max_allowed_error=0.15)
        print(validation.summary())
        check("Validation ran", validation.total_tensors > 0)


def test_footer_crc_validation() -> None:
    """Test footer CRC32 validation against file content."""
    print("\n=== Footer CRC Validation ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "crc_test.sscx")
        writer = SSCXWriter(path, model_name="crc-test")
        t = np.ones(16, dtype=np.float32)
        writer.add_tensor("t", t)
        writer.save()

        # Read and verify footer CRC
        with open(path, "rb") as f:
            data = f.read()
        footer = SSCXFooter.unpack(data, len(data) - 12)
        check("Footer CRC matches", footer.crc32 == _crc32(data[:-12]))
        check("Footer tensor_index_offset > 0", footer.tensor_index_offset > 0)
        check("Footer layer_index_offset > 0", footer.layer_index_offset > 0)


if __name__ == "__main__":
    print("=" * 60)
    print("SSCX Format & ModelCompressor Test Suite")
    print("=" * 60)

    test_utilities()
    test_header_pack_unpack()
    test_header_bad_magic()
    test_tensor_info_pack_unpack()
    test_layer_info_pack_unpack()
    test_footer_pack_unpack()
    test_crc32()
    test_zigzag()
    test_error_metrics()
    test_sensitivity()
    test_layer_detection()
    test_int8_compress_decompress()
    test_int4_compress_decompress()
    test_dct_compress_decompress()
    test_write_read_roundtrip()
    test_write_read_precompressed()
    test_footer_crc_validation()
    test_synthetic_model_compress()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        exit(1)

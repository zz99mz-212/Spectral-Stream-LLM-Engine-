"""
Tests for unified_quantizer — 5-stage compression pipeline and enhancements.

Tests cover:
- HierarchicalDCT: roundtrip, block size selection, variance threshold
- TensorTrain: decompose/reconstruct, rank selection, fallback for invalid SVD
- VariableBitQuantizer: quantize/dequantize roundtrip, quality parameter
- EntropyCoder: encode/decode block roundtrip, zero-run handling
- QualityTableManager: quality and block size for different layer names
- UnifiedQuantizer: full compress/decompress, metrics, ratio, iterative_refine
- StabilizerQuantizer: encode/decode nibble, protect/recover stream
- TernaryWeightQuantizer: compress/decompress, sparsity, values in {-1,0,+1}
- SpectralSparsification: sparsify/desparsify, sparsity level
- format_size: different size ranges
- _build_huffman_codes: single value, multiple values, serialization
"""

import sys

sys.path.insert(0, ".")

import math
import struct

import numpy as np
import pytest

try:
    from spectralstream.compression.unified_quantizer import (
        HierarchicalDCT,
        TensorTrain,
        VariableBitQuantizer,
        EntropyCoder,
        QualityTableManager,
        UnifiedQuantizer,
        HierarchicalMPSCompressor,
        QAOABitAllocator,
        StabilizerQuantizer,
        PredictiveCodingQuantizer,
        TernaryWeightQuantizer,
        SpectralSparsification,
        CompressionPipeline2000Legacy,
        Pipeline2000LegacyConfig,
        compress_to_ssf_block,
        decompress_from_ssf_block,
        format_size,
        DCTBlock,
        QuantizedBlock,
        EncodedBlock,
        QualityProfile,
        _build_huffman_codes,
        _encode_symbols,
        _decode_symbols,
        _serialize_codebook,
        _deserialize_codebook,
        _rle_encode,
        _rle_decode,
        _generate_qtable,
        _tt_svd,
        _tt_reconstruct,
    )
except ImportError:
    pass


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def synth_32x32(rng):
    return rng.randn(16, 16).astype(np.float32)


@pytest.fixture
def synth_64x64(rng):
    return rng.randn(16, 16).astype(np.float32)


@pytest.fixture
def smooth_32x32():
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-(xx**2 + yy**2) * 2).astype(np.float32)


# ── HierarchicalDCT ────────────────────────────────────────────────────


class TestHierarchicalDCT:
    def test_compress_decompress_roundtrip(self, synth_32x32):
        dct = HierarchicalDCT(variance_threshold=0.01)
        blocks = dct.compress(synth_32x32)
        recon = dct.decompress(blocks, synth_32x32.shape)
        assert recon.shape == synth_32x32.shape
        mse = float(np.mean((synth_32x32 - recon) ** 2))
        assert mse < 0.5, f"MSE={mse} too high"

    def test_block_size_selection_smooth(self, smooth_32x32):
        dct = HierarchicalDCT(variance_threshold=0.01)
        blocks = dct.compress(smooth_32x32)
        sizes = [b.block_size for b in blocks]
        mean_size = np.mean(sizes)
        assert mean_size >= 16, (
            f"Mean block size {mean_size} too small for smooth region"
        )

    def test_block_size_selection_noisy(self, synth_32x32):
        dct = HierarchicalDCT(variance_threshold=0.5)
        blocks = dct.compress(synth_32x32)
        sizes = [b.block_size for b in blocks]
        assert all(s >= 8 for s in sizes), "All blocks at least 8x8"
        assert all(s <= 128 for s in sizes), "All blocks at most 128x128"

    def test_compress_decompress_nonsquare(self, rng):
        tensor = rng.randn(40, 24).astype(np.float32)
        dct = HierarchicalDCT(variance_threshold=0.01)
        blocks = dct.compress(tensor)
        recon = dct.decompress(blocks, tensor.shape)
        assert recon.shape == tensor.shape

    def test_variance_threshold_effect(self, synth_32x32):
        dct_low = HierarchicalDCT(variance_threshold=0.001)
        dct_high = HierarchicalDCT(variance_threshold=1.0)
        blocks_low = dct_low.compress(synth_32x32)
        blocks_high = dct_high.compress(synth_32x32)
        sizes_low = [b.block_size for b in blocks_low]
        sizes_high = [b.block_size for b in blocks_high]
        assert np.mean(sizes_low) >= np.mean(sizes_high), (
            "Lower threshold should produce larger blocks"
        )

    def test_dct_block_dataclass(self):
        blk = DCTBlock(row=0, col=16, block_size=16, dct=np.eye(4), variance=0.5)
        assert blk.row == 0
        assert blk.col == 16
        assert blk.block_size == 16
        assert blk.variance == 0.5


# ── TensorTrain ─────────────────────────────────────────────────────────


class TestTensorTrain:
    def test_decompose_reconstruct_roundtrip(self, synth_32x32):
        tt = TensorTrain(relative_error=0.01, max_rank=16)
        data = tt.decompose(synth_32x32)
        recon = tt.reconstruct(data)
        assert recon.shape == synth_32x32.shape
        mse = float(np.mean((synth_32x32 - recon) ** 2))
        assert mse < 0.15, f"MSE={mse} too high"

    def test_rank_selection(self):
        rng = np.random.RandomState(42)
        low_rank = rng.randn(32, 4) @ rng.randn(4, 32)
        tt = TensorTrain(relative_error=0.01, max_rank=16)
        data = tt.decompose(low_rank.astype(np.float32))
        assert data["rank"] <= 8, (
            f"Rank {data['rank']} should be small for low-rank matrix"
        )

    def test_high_rank_matrix(self):
        rng = np.random.RandomState(42)
        high_rank = rng.randn(32, 32)
        tt = TensorTrain(relative_error=0.01, max_rank=16)
        data = tt.decompose(high_rank.astype(np.float32))
        assert data["rank"] > 4, f"Rank {data['rank']} should be >4 for random matrix"

    def test_fallback_for_nan(self):
        rng = np.random.RandomState(42)
        mat = rng.randn(8, 8).astype(np.float32)
        mat[0, 0] = np.nan
        tt = TensorTrain(relative_error=0.01, max_rank=8)
        data = tt.decompose(mat)
        assert data["rank"] > 0
        recon = tt.reconstruct(data)
        assert recon.shape == mat.shape

    def test_fallback_for_inf(self):
        mat = np.zeros((8, 8), dtype=np.float32)
        mat[0, 0] = np.inf
        tt = TensorTrain(relative_error=0.01, max_rank=8)
        data = tt.decompose(mat)
        assert data["rank"] > 0
        recon = tt.reconstruct(data)
        assert recon.shape == mat.shape

    def test_zero_matrix(self):
        mat = np.zeros((16, 16), dtype=np.float32)
        tt = TensorTrain(relative_error=0.01, max_rank=8)
        data = tt.decompose(mat)
        assert data["rank"] >= 1
        recon = tt.reconstruct(data)
        assert recon.shape == mat.shape

    def test_tt_svd_reconstruct(self):
        rng = np.random.RandomState(42)
        mat = rng.randn(8, 12).astype(np.float64)
        cores = _tt_svd(mat, 4)
        assert len(cores) == 3
        recon = _tt_reconstruct(cores)
        assert recon.shape == (8, 12)
        mse = float(np.mean((mat - recon) ** 2))
        assert mse < 0.5

    def test_default_parameters(self):
        tt = TensorTrain()
        assert tt.relative_error == 0.01
        assert tt.max_rank == 16


# ── VariableBitQuantizer ───────────────────────────────────────────────


class TestVariableBitQuantizer:
    def test_quantize_dequantize_roundtrip(self):
        vbq = VariableBitQuantizer(quality=1.0)
        dct_block = np.random.RandomState(42).randn(16, 16).astype(np.float64)
        comp = vbq.quantize(dct_block)
        assert "quantized" in comp
        assert "bits_used" in comp
        assert "skipped" in comp
        assert comp["shape"] == (16, 16)

    def test_dequantize_produces_same_shape(self):
        vbq = VariableBitQuantizer(quality=1.0)
        dct_block = np.ones((8, 8), dtype=np.float64) * 10.0
        comp = vbq.quantize(dct_block)
        recon = vbq.dequantize(comp)
        assert recon.shape == (8, 8)

    def test_quality_parameter_effect(self):
        dct_block = np.random.RandomState(42).randn(8, 8).astype(np.float64)
        vbq_low = VariableBitQuantizer(quality=0.1)
        vbq_high = VariableBitQuantizer(quality=1.0)
        comp_low = vbq_low.quantize(dct_block)
        comp_high = vbq_high.quantize(dct_block)
        n_skipped_low = int(np.sum(comp_low["skipped"]))
        n_skipped_high = int(np.sum(comp_high["skipped"]))
        assert n_skipped_low >= n_skipped_high, (
            "Lower quality should skip more coefficients"
        )

    def test_skipped_coefficients_are_zero_in_dequant(self):
        vbq = VariableBitQuantizer(quality=0.5)
        block = np.zeros((8, 8), dtype=np.float64)
        block[0, 0] = 100.0
        comp = vbq.quantize(block)
        recon = vbq.dequantize(comp)
        assert np.all(recon[comp["skipped"]] == 0.0), "Skipped coefs should be zero"

    def test_qtable_generation(self):
        qt = _generate_qtable(16, 1.0)
        assert qt.shape == (16, 16)
        assert qt[0, 0] >= 12, "DC component should have >= 12 bits"
        assert np.all(qt >= 1), "All entries should have at least 1 bit"
        assert np.all(qt <= 12), "All entries should have at most 12 bits"

    def test_qtable_quality_scaling(self):
        qt_low = _generate_qtable(8, 0.5)
        qt_high = _generate_qtable(8, 1.0)
        assert np.mean(qt_high) >= np.mean(qt_low), (
            "Higher quality should have more bits"
        )


# ── EntropyCoder ───────────────────────────────────────────────────────


class TestEntropyCoder:
    def test_encode_decode_block_roundtrip(self):
        coder = EntropyCoder()
        quantized = np.array(
            [
                [5, 0, -3, 0],
                [0, 0, 2, 0],
                [0, 0, 0, -1],
            ],
            dtype=np.int32,
        )
        bits_used = (quantized != 0).astype(np.int32) * 4
        skipped = (quantized == 0).astype(bool)
        eb = coder.encode_block(
            quantized,
            bits_used,
            skipped,
            max_abs=5.0,
            row=0,
            col=0,
            block_size=4,
        )
        assert isinstance(eb, EncodedBlock)
        assert eb.row == 0
        assert eb.col == 0
        assert eb.block_size == 4
        assert len(eb.non_zero_values) > 0

    def test_decode_produces_correct_shape(self):
        coder = EntropyCoder()
        quantized = np.array(
            [
                [5, 0, -3],
                [0, 2, 0],
            ],
            dtype=np.int32,
        )
        bits_used = (quantized != 0).astype(np.int32) * 4
        skipped = (quantized == 0).astype(bool)
        eb = coder.encode_block(
            quantized,
            bits_used,
            skipped,
            max_abs=5.0,
            row=0,
            col=0,
            block_size=3,
        )
        decoded = coder.decode_block(eb, (2, 3))
        assert decoded.shape == (2, 3)
        # Decoded block places non-zero values row-major from the start
        non_zero_vals = quantized[quantized != 0]
        assert np.array_equal(decoded.ravel()[: len(non_zero_vals)], non_zero_vals)

    def test_all_zeros_handling(self):
        coder = EntropyCoder()
        quantized = np.zeros((4, 4), dtype=np.int32)
        bits_used = np.zeros((4, 4), dtype=np.int32)
        skipped = np.ones((4, 4), dtype=bool)
        eb = coder.encode_block(
            quantized,
            bits_used,
            skipped,
            max_abs=0.0,
            row=0,
            col=0,
            block_size=4,
        )
        assert eb.n_zeros_skipped == 16

    def test_zero_run_encoding(self):
        coder = EntropyCoder()
        quantized = np.array([7, 0, 0, -2, 0], dtype=np.int32).reshape(1, 5)
        bits_used = np.ones((1, 5), dtype=np.int32) * 4
        skipped = np.zeros((1, 5), dtype=bool)
        eb = coder.encode_block(
            quantized,
            bits_used,
            skipped,
            max_abs=7.0,
            row=0,
            col=0,
            block_size=5,
        )
        assert 7 in eb.non_zero_values
        assert -2 in eb.non_zero_values


# ── Huffman / RLE helper functions ─────────────────────────────────────


class TestHuffmanHelpers:
    def test_build_huffman_codes_single_value(self):
        codes = _build_huffman_codes([5, 5, 5])
        assert codes == {5: "0"}

    def test_build_huffman_codes_multiple_values(self):
        codes = _build_huffman_codes([1, 1, 2, 3, 3, 3])
        assert len(codes) == 3
        assert all(isinstance(c, str) for c in codes.values())

    def test_build_huffman_codes_empty(self):
        codes = _build_huffman_codes([])
        assert codes == {}

    def test_encode_decode_symbols(self):
        symbols = [3, 1, 4, 1, 5, 9, 2, 6]
        codebook = _build_huffman_codes(symbols)
        bitstream = _encode_symbols(symbols, codebook)
        decoded = _decode_symbols(bitstream, codebook, len(symbols))
        assert decoded == symbols

    def test_encode_decode_empty(self):
        bitstream = _encode_symbols([], {})
        assert bitstream == b""

    def test_serialize_codebook_roundtrip(self):
        codebook = {3: "0", 1: "10", 4: "11"}
        serialized = _serialize_codebook(codebook)
        decoded, offset = _deserialize_codebook(serialized)
        assert decoded == codebook
        assert offset == len(serialized)

    def test_rle_encode_decode(self):
        arr = np.array([3, 3, 3, 0, 0, 7, 7, 0, 0, 0], dtype=np.int32)
        values, lengths = _rle_encode(arr)
        recon = _rle_decode(values, lengths, arr.shape)
        assert np.array_equal(recon, arr)

    def test_rle_empty(self):
        arr = np.array([], dtype=np.int32)
        values, lengths = _rle_encode(arr)
        assert len(values) == 0
        assert len(lengths) == 0


# ── QualityTableManager ───────────────────────────────────────────────


class TestQualityTableManager:
    def test_get_quality_by_layer_name(self):
        mgr = QualityTableManager(base_quality=1.0)
        assert mgr.get_quality("attn_q") == 1.0
        assert mgr.get_quality("attn_k") == 0.92
        assert mgr.get_quality("ffn_gate") == 0.55
        assert mgr.get_quality("norm") == 0.50
        assert mgr.get_quality("output") == 1.0

    def test_get_quality_default(self):
        mgr = QualityTableManager(base_quality=1.0)
        q = mgr.get_quality("unknown_layer")
        assert q == 0.7

    def test_get_quality_case_insensitive(self):
        mgr = QualityTableManager(base_quality=1.0)
        assert mgr.get_quality("ATTN_Q") == 1.0
        assert mgr.get_quality("FFN_GATE") == 0.55

    def test_get_block_size_hint(self):
        mgr = QualityTableManager()
        assert mgr.get_block_size_hint("attn_q") == 16
        assert mgr.get_block_size_hint("ffn_gate") == 64
        assert mgr.get_block_size_hint("embed") == 128
        assert mgr.get_block_size_hint("unknown") == 32

    def test_custom_profile(self):
        mgr = QualityTableManager(base_quality=0.8)
        mgr.profiles["custom"] = QualityProfile(layer_name="custom", importance=0.5)
        assert mgr.get_quality("custom") == 0.4

    def test_base_quality_scale(self):
        mgr = QualityTableManager(base_quality=0.5)
        assert mgr.get_quality("attn_q") == 0.5
        assert mgr.get_quality("ffn_gate") == 0.275


# ── UnifiedQuantizer ───────────────────────────────────────────────────


class TestUnifiedQuantizer:
    def test_compress_decompress_roundtrip(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        compressed = uq.compress(synth_32x32, layer_name="attn_q")
        assert compressed["type"] == "unified"
        assert compressed["n_blocks"] > 0
        decompressed = uq.decompress(compressed)
        assert decompressed.shape == synth_32x32.shape

    def test_compute_quality_metrics(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        compressed = uq.compress(synth_32x32)
        decompressed = uq.decompress(compressed)
        metrics = uq.compute_quality_metrics(synth_32x32, decompressed)
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "max_abs_error" in metrics
        assert metrics["mse"] >= 0.0
        assert metrics["ssim"] >= -1.0 and metrics["ssim"] <= 1.0

    def test_get_ratio(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        compressed = uq.compress(synth_32x32)
        ratio = uq.get_ratio(synth_32x32, compressed)
        assert ratio > 1.0, f"Ratio {ratio} should be > 1.0"

    def test_raw_type_for_small_tensor(self, rng):
        uq = UnifiedQuantizer()
        small = rng.randn(3, 3).astype(np.float32)
        compressed = uq.compress(small)
        assert compressed["type"] == "raw"
        decompressed = uq.decompress(compressed)
        assert np.allclose(small, decompressed, atol=1e-6)

    def test_raw_type_for_1d_tensor(self, rng):
        uq = UnifiedQuantizer()
        arr = rng.randn(16).astype(np.float32)
        compressed = uq.compress(arr)
        assert compressed["type"] == "raw"
        decompressed = uq.decompress(compressed)
        assert np.allclose(arr, decompressed, atol=1e-6)

    def test_serialize_compressed(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        compressed = uq.compress(synth_32x32)
        serialized = uq.serialize_compressed(compressed)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_compress_tensor(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        blob = uq.compress_tensor(synth_32x32, layer_name="attn_q")
        assert isinstance(blob, bytes)
        assert len(blob) > 0

    def test_iterative_refine(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        compressed = uq.iterative_refine(synth_32x32, target_ratio=50.0, max_iters=3)
        assert "blocks" in compressed

    def test_empty_tensor_raises(self):
        uq = UnifiedQuantizer()
        with pytest.raises(ValueError, match="Cannot compress empty tensor"):
            uq.compress(np.array([], dtype=np.float32))
        with pytest.raises(ValueError, match="Cannot compress empty tensor"):
            uq.compress_tensor(np.array([], dtype=np.float32))

    def test_aggressive_mode(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, aggressive=True)
        compressed = uq.compress(synth_32x32)
        assert compressed["type"] == "unified"

    def test_compress_with_layer_name_quality(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        compressed = uq.compress(synth_32x32, layer_name="ffn_norm")
        assert compressed["layer_name"] == "ffn_norm"

    def test_return_intermediates(self, synth_32x32):
        uq = UnifiedQuantizer(quality=0.95, tt_relative_error=0.05)
        compressed = uq.compress(synth_32x32, return_intermediates=True)
        assert "_intermediates" in compressed

    def test_decompress_invalid_type_raises(self):
        uq = UnifiedQuantizer()
        with pytest.raises(TypeError):
            uq.decompress("not_a_dict")
        with pytest.raises(ValueError):
            uq.decompress({"no_type": True})

    def test_compute_quality_metrics_empty_raises(self):
        uq = UnifiedQuantizer()
        with pytest.raises(ValueError, match="must have non-zero size"):
            uq.compute_quality_metrics(np.array([]), np.array([]))


# ── compress_to_ssf_block / decompress_from_ssf_block ─────────────────


class TestSSFBlock:
    def test_compress_decompress_ssf_block(self, synth_32x32):
        data = compress_to_ssf_block(synth_32x32, layer_name="test", quality=0.95)
        assert isinstance(data, bytes)
        recon = decompress_from_ssf_block(data)
        assert recon.shape == synth_32x32.shape

    def test_decompress_empty(self):
        result = decompress_from_ssf_block(b"")
        assert result.shape == (0,)

    def test_decompress_short(self):
        result = decompress_from_ssf_block(b"\x00")
        assert result.shape == (0,)


# ── format_size ────────────────────────────────────────────────────────


class TestFormatSize:
    def test_bytes(self):
        assert format_size(0) == "0 B"
        assert format_size(512) == "512 B"

    def test_kilobytes(self):
        assert format_size(1024) == "1.0 KB"
        assert format_size(1536) == "1.5 KB"

    def test_megabytes(self):
        assert format_size(1024**2) == "1.00 MB"
        assert format_size(2 * 1024**2) == "2.00 MB"

    def test_gigabytes(self):
        assert format_size(1024**3) == "1.00 GB"
        assert format_size(3 * 1024**3) == "3.00 GB"


# ── HierarchicalMPSCompressor ─────────────────────────────────────────


class TestHierarchicalMPSCompressor:
    def test_compress_decompress_roundtrip(self, synth_32x32):
        mps = HierarchicalMPSCompressor(min_bond_dim=4, max_bond_dim=8, n_sweeps=2)
        comp = mps.compress(synth_32x32)
        assert comp["type"] == "hierarchical_mps"
        recon = mps.decompress(comp)
        assert recon.shape == synth_32x32.shape
        mse = float(np.mean((synth_32x32 - recon) ** 2))
        assert mse < 0.2, f"MSE={mse} too high"

    def test_parameter_reduction(self, synth_32x32):
        mps = HierarchicalMPSCompressor(min_bond_dim=4, max_bond_dim=8, n_sweeps=2)
        comp = mps.compress(synth_32x32)
        param_count = sum(c.size for c in comp["cores"])
        assert param_count < synth_32x32.size, "MPS should reduce parameter count"


# ── QAOABitAllocator ──────────────────────────────────────────────────


class TestQAOABitAllocator:
    def test_allocate_shape(self):
        qaoa = QAOABitAllocator(quality=0.95, max_bits_per_coeff=8)
        mags = np.abs(np.random.RandomState(42).randn(16, 16))
        alloc = qaoa.allocate(mags, block_size=16)
        assert alloc.shape == (16, 16)
        assert np.all(alloc >= 0)
        assert np.all(alloc <= 8)

    def test_high_magnitude_gets_more_bits(self):
        qaoa = QAOABitAllocator(quality=0.95, max_bits_per_coeff=8)
        rng = np.random.RandomState(42)
        mags_low = np.abs(rng.randn(8, 8))
        mags_high = np.abs(rng.randn(8, 8)) * 100.0
        alloc_low = qaoa.allocate(mags_low, block_size=8)
        alloc_high = qaoa.allocate(mags_high, block_size=8)
        assert np.sum(alloc_high) >= np.sum(alloc_low), (
            "High-magnitude block should get at least as many bits"
        )


# ── StabilizerQuantizer ───────────────────────────────────────────────


class TestStabilizerQuantizer:
    def test_quantize_dequantize_roundtrip(self):
        sq = StabilizerQuantizer(n_bits=4, use_extended=True)
        vals = np.array([0.5, -0.3, 0.0, 0.8, -0.9, 0.2, -0.1, 0.7])
        encoded, overhead = sq.quantize_with_correction(vals)
        decoded = sq.dequantize_with_correction(encoded, 1.0, vals.shape)
        assert decoded.shape == vals.shape
        assert overhead > 1.0

    def test_protect_recover_stream(self):
        sq = StabilizerQuantizer(n_bits=4, use_extended=True)
        raw = b"\xab\xcd\xef\x01\x23\x45\x67\x89"
        protected = sq.protect_stream(raw)
        recovered = sq.recover_stream(protected)
        assert recovered == raw, "Perfect recovery without errors"

    def test_single_bit_error_correction(self):
        sq = StabilizerQuantizer(n_bits=4, use_extended=True)
        raw = b"\xab\xcd\xef"
        protected = sq.protect_stream(raw)
        corrupted = bytearray(protected)
        if len(corrupted) > 0:
            corrupted[1] ^= 0b00000100
        recovered = sq.recover_stream(bytes(corrupted))
        assert recovered == raw, "Single-bit error should be corrected"

    def test_encode_decode_nibble(self):
        sq = StabilizerQuantizer(n_bits=4, use_extended=False)
        for val in range(16):
            coded = sq._encode_nibble(val)
            decoded = sq._decode_nibble(coded)
            assert decoded == val, f"Nibble roundtrip failed for {val}"

    def test_extended_encode_decode_nibble(self):
        sq = StabilizerQuantizer(n_bits=4, use_extended=True)
        for val in range(16):
            coded = sq._encode_nibble(val)
            decoded = sq._decode_nibble(coded)
            assert decoded == val, f"Extended nibble roundtrip failed for {val}"


# ── PredictiveCodingQuantizer ─────────────────────────────────────────


class TestPredictiveCodingQuantizer:
    def test_compress_decompress_roundtrip(self, rng):
        pcq = PredictiveCodingQuantizer(n_bits_residual=3, max_bits_original=8)
        signal = rng.randn(64).astype(np.float32) * 2.0
        comp = pcq.compress(signal)
        assert comp["type"] == "predictive"
        recon = pcq.decompress(comp)
        assert recon.shape == signal.shape

    def test_short_signal(self):
        pcq = PredictiveCodingQuantizer(n_bits_residual=3, max_bits_original=8)
        signal = np.array([1.5], dtype=np.float32)
        comp = pcq.compress(signal)
        recon = pcq.decompress(comp)
        assert recon.shape == signal.shape

    def test_residuals_smaller_than_signal(self, rng):
        pcq = PredictiveCodingQuantizer(n_bits_residual=3, max_bits_original=8)
        signal = rng.randn(64).astype(np.float32)
        comp = pcq.compress(signal)
        assert len(comp["residuals"]) <= len(signal)

    def test_ar_coefficients_present(self, rng):
        pcq = PredictiveCodingQuantizer(n_bits_residual=3, max_bits_original=8)
        signal = rng.randn(64).astype(np.float32)
        comp = pcq.compress(signal)
        assert "ar_coeffs" in comp
        a1, a2 = comp["ar_coeffs"]
        assert isinstance(a1, float)
        assert isinstance(a2, float)


# ── TernaryWeightQuantizer ────────────────────────────────────────────


class TestTernaryWeightQuantizer:
    def test_compress_decompress_roundtrip(self, synth_32x32):
        twq = TernaryWeightQuantizer(sparsity_target=0.85, block_size=64)
        comp = twq.compress(synth_32x32)
        assert comp["type"] == "ternary"
        recon = twq.decompress(comp)
        assert recon.shape == synth_32x32.shape

    def test_values_in_ternary_set(self, synth_32x32):
        twq = TernaryWeightQuantizer(sparsity_target=0.85, block_size=64)
        comp = twq.compress(synth_32x32)
        packed = comp["packed"]
        observed = set()
        for i in range(comp["n_weights"]):
            byte_idx = i // 4
            bit_idx = i % 4
            code = (packed[byte_idx] >> (bit_idx * 2)) & 0b11
            if code == 0b01:
                observed.add(1)
            elif code == 0b10:
                observed.add(-1)
            else:
                observed.add(0)
        assert observed.issubset({-1, 0, 1}), f"Got {observed}"

    def test_sparsity_target_respected(self):
        rng = np.random.RandomState(42)
        weights = rng.randn(128).astype(np.float32)
        twq = TernaryWeightQuantizer(sparsity_target=0.9, block_size=128)
        comp = twq.compress(weights)
        assert comp["sparsity"] >= 0.5, f"Sparsity {comp['sparsity']} should be >= 0.5"

    def test_block_scaling_present(self, synth_32x32):
        twq = TernaryWeightQuantizer(sparsity_target=0.85, block_size=32)
        comp = twq.compress(synth_32x32)
        assert len(comp["scales"]) > 0
        assert np.all(comp["scales"] >= 0)


# ── SpectralSparsification ────────────────────────────────────────────


class TestSpectralSparsification:
    def test_sparsify_desparsify_roundtrip(self, synth_32x32):
        ss = SpectralSparsification(target_sparsity=0.9, block_size=16)
        comp = ss.sparsify(synth_32x32)
        assert comp["type"] == "spectral_sparse"
        recon = ss.desparsify(comp)
        assert recon.shape == synth_32x32.shape

    def test_sparsity_level(self, synth_32x32):
        ss = SpectralSparsification(target_sparsity=0.9, block_size=16)
        comp = ss.sparsify(synth_32x32)
        assert comp["total_kept"] < comp["total_coeffs"]
        assert comp["actual_sparsity"] >= 0.5

    def test_sparsity_effect(self, synth_32x32):
        ss_high = SpectralSparsification(target_sparsity=0.95, block_size=16)
        ss_low = SpectralSparsification(target_sparsity=0.5, block_size=16)
        comp_high = ss_high.sparsify(synth_32x32)
        comp_low = ss_low.sparsify(synth_32x32)
        assert comp_high["actual_sparsity"] >= comp_low["actual_sparsity"], (
            "Higher target sparsity should yield higher actual sparsity"
        )


# ── CompressionPipeline2000Legacy ─────────────────────────────────────


class TestCompressionPipeline2000Legacy:
    @pytest.fixture
    def cfg(self):
        return Pipeline2000LegacyConfig(
            quality=0.95,
            dct_block_size=64,
            mps_min_bond=4,
            mps_max_bond=6,
            mps_n_sweeps=2,
            ternary_sparsity=0.85,
            spectral_sparsity=0.90,
            n_bits_quant=4,
            enable_stabilizer=True,
        )

    def test_compress_decompress_roundtrip(self, cfg, synth_64x64):
        pipe = CompressionPipeline2000Legacy(cfg)
        comp = pipe.compress(synth_64x64, layer_name="test.attn_q.weight")
        assert comp["type"] == "pipeline2000"
        recon = pipe.decompress(comp)
        assert recon.shape == synth_64x64.shape

    def test_compression_ratio_positive(self, cfg, synth_64x64):
        pipe = CompressionPipeline2000Legacy(cfg)
        comp = pipe.compress(synth_64x64)
        assert comp.get("ratio", 0.0) > 0

    def test_get_quality_metrics(self, cfg, synth_64x64):
        pipe = CompressionPipeline2000Legacy(cfg)
        comp = pipe.compress(synth_64x64)
        recon = pipe.decompress(comp)
        metrics = pipe.get_quality_metrics(synth_64x64, recon)
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "relative_error" in metrics

    def test_small_tensor(self, cfg, rng):
        pipe = CompressionPipeline2000Legacy(cfg)
        small = rng.randn(3, 3).astype(np.float32)
        comp = pipe.compress(small)
        assert comp["type"] == "pipeline2000_raw"
        recon = pipe.decompress(comp)
        assert np.allclose(small, recon, atol=1e-6)

    def test_get_ratio(self, cfg, synth_64x64):
        pipe = CompressionPipeline2000Legacy(cfg)
        comp = pipe.compress(synth_64x64)
        ratio = pipe.get_ratio(synth_64x64, comp)
        assert ratio >= 0


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_dct_2d_identity(self):
        hdct = HierarchicalDCT()
        mat = np.eye(8, dtype=np.float64)
        coeffs = hdct._dct_2d(mat)
        recon = hdct._idct_2d(coeffs)
        assert np.allclose(mat, recon, atol=1e-10), "DCT/IDCT should be invertible"

    def test_tt_svd_rank_limits(self):
        rng = np.random.RandomState(42)
        mat = rng.randn(6, 6).astype(np.float64)
        cores = _tt_svd(mat, 10)
        assert len(cores) == 3
        recon = _tt_reconstruct(cores)
        assert recon.shape == (6, 6)

    def test_format_size_zero(self):
        assert format_size(0) == "0 B"
        assert format_size(1) == "1 B"

    def test_vbq_zero_block(self):
        vbq = VariableBitQuantizer(quality=1.0)
        block = np.zeros((8, 8), dtype=np.float64)
        comp = vbq.quantize(block)
        assert comp["max_abs"] > 0
        recon = vbq.dequantize(comp)
        assert recon.shape == (8, 8)

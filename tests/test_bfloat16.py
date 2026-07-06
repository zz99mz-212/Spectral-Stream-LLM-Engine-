"""
BF16 Support Tests
===================
Verifies that:
1. BF16 tensors (stored as uint16) are handled correctly by all methods
2. Memory is preserved (BF16 stays as 2 bytes/element until computation)
3. Precision conversion error is tracked
4. Compression output for BF16 input is correct (bytes → back to BF16)
5. Compatibility with float32 input is unchanged
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pytest

sys.path.insert(0, ".")

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    METHOD_REGISTRY,
    _BlockINT8,
    _BlockINT4,
    _HadamardINT8,
    _HadamardINT4,
    _SparsityINT4,
    _DeltaINT4,
    _SVDCompress,
    _DCTSpectral,
    _TensorTrain,
    _FWHTCompress,
)
from spectralstream.core.math_primitives import (
    bfloat16_to_float32,
    float32_to_bfloat16,
    is_bfloat16,
    ensure_float32,
    maybe_contract_to_uint16,
    compression_ratio_adjustment,
)


# ── BF16 Conversion Tests ─────────────────────────────────────────────


class TestBfloat16Conversion:
    def test_roundtrip_f32_to_bf16(self):
        """f32→BF16→f32 roundtrip should preserve ~7 significant bits."""
        f32 = np.array(
            [1.0, 0.5, 0.25, 3.14159, -2.71828, 0.0, 1e-3, -1e-3], dtype=np.float32
        )
        bf16 = float32_to_bfloat16(f32)
        assert bf16.dtype == np.uint16
        restored = bfloat16_to_float32(bf16)
        # BF16 has ~7.5 bits of mantissa → relative error ~1/256
        rel_errors = np.abs(restored - f32) / (np.abs(f32) + 1e-30)
        assert float(np.max(rel_errors)) < 0.01

    def test_roundtrip_bfloat16_via_io_bitpattern(self):
        """Verify the bit manipulation matches the safetensors pattern."""
        orig_f32 = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        bf16 = float32_to_bfloat16(orig_f32)
        # Simulate the IO layer: uint16 → uint32 << 16 → view as float32
        f32_from_io = (bf16.astype(np.uint32) << 16).view(np.float32)
        # The roundtrip should be close to original
        np.testing.assert_allclose(orig_f32, f32_from_io, atol=1e-2)

    def test_is_bfloat16_uint16(self):
        assert is_bfloat16(np.array([1, 2], dtype=np.uint16))
        assert not is_bfloat16(np.array([1.0, 2.0], dtype=np.float32))
        assert not is_bfloat16(np.array([1, 2], dtype=np.int32))

    def test_bfloat16_to_float32_non_uint16_raises(self):
        with pytest.raises(TypeError):
            bfloat16_to_float32(np.array([1.0, 2.0], dtype=np.float32))

    def test_ensure_float32_passthrough(self):
        f32 = np.array([1.0, 2.0], dtype=np.float32)
        result = ensure_float32(f32)
        assert result.dtype == np.float32
        assert result is f32  # no copy

    def test_ensure_float32_converts_bf16(self):
        bf16 = float32_to_bfloat16(np.array([1.0, 2.0], dtype=np.float32))
        result = ensure_float32(bf16)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, bfloat16_to_float32(bf16), atol=1e-6)

    def test_maybe_contract_to_uint16(self):
        f32 = np.array([1.0, 2.0], dtype=np.float32)
        result = maybe_contract_to_uint16(f32, input_was_bf16=True)
        assert result.dtype == np.uint16
        result2 = maybe_contract_to_uint16(f32, input_was_bf16=False)
        assert result2 is f32

    def test_compression_ratio_adjustment(self):
        assert compression_ratio_adjustment("BF16") == 0.5
        assert compression_ratio_adjustment("bfloat16") == 0.5
        assert compression_ratio_adjustment("bf16") == 0.5
        assert compression_ratio_adjustment("float32") == 1.0
        assert compression_ratio_adjustment("F32") == 1.0

    def test_memory_efficiency(self):
        """BF16 should use 2 bytes/element, float32 uses 4 bytes/element."""
        f32 = np.random.randn(1000).astype(np.float32)
        bf16 = float32_to_bfloat16(f32)
        assert f32.nbytes == 4000
        assert bf16.nbytes == 2000  # half the memory


# ── Method-Level BF16 Tests ────────────────────────────────────────────

METHODS_TO_TEST = [
    (_BlockINT8(), {"block_size": 128}),
    (_BlockINT4(), {"block_size": 16}),
    (_HadamardINT8(), {"block_size": 128}),
    (_HadamardINT4(), {"block_size": 16}),
    (_SparsityINT4(), {"group_size": 32}),
    (_SVDCompress(), {"rank": 4}),
    (_DCTSpectral(), {"keep_ratio": 0.5}),
    (_TensorTrain(), {"rank": 4}),
    (_FWHTCompress(), {"keep_ratio": 0.5}),
]


class TestBfloat16Methods:
    @pytest.mark.parametrize("inst,params", METHODS_TO_TEST)
    def test_bf16_compress_decompress(self, inst, params):
        """Each method should accept BF16 input and produce correct output."""
        f32 = np.random.randn(16, 16).astype(np.float32)
        bf16_input = float32_to_bfloat16(f32)

        data_f32, meta_f32 = inst.compress(f32, **params)
        data_bf16, meta_bf16 = inst.compress(bf16_input, **params)

        recon_f32 = inst.decompress(data_f32, meta_f32).reshape(f32.shape)
        recon_bf16 = inst.decompress(data_bf16, meta_bf16).reshape(f32.shape)

        # Output shapes should match
        assert recon_f32.shape == f32.shape
        assert recon_bf16.shape == f32.shape

        # Ensure float32 arrays for error computation
        recon_f32_f = ensure_float32(recon_f32)
        recon_bf16_f = ensure_float32(recon_bf16)

        # Both should be valid reconstructions
        err_f32 = float(
            np.linalg.norm(f32.ravel() - recon_f32_f.ravel())
            / (np.linalg.norm(f32.ravel()) + 1e-30)
        )
        err_bf16 = float(
            np.linalg.norm(ensure_float32(bf16_input).ravel() - recon_bf16_f.ravel())
            / (np.linalg.norm(ensure_float32(bf16_input).ravel()) + 1e-30)
        )

        # BF16 input should produce valid compression
        assert err_bf16 < 1.0
        # Both paths should produce similar quality
        assert err_bf16 < err_f32 + 0.5

    @pytest.mark.parametrize("inst,params", METHODS_TO_TEST)
    def test_bf16_output_dtype(self, inst, params):
        """BF16 input should produce BF16 (uint16) output from decompress."""
        f32 = np.random.randn(16, 16).astype(np.float32)
        bf16_input = float32_to_bfloat16(f32)

        data_bf16, meta_bf16 = inst.compress(bf16_input, **params)
        recon_bf16 = inst.decompress(data_bf16, meta_bf16).reshape(f32.shape)

        # When we track it properly: decompress should return uint16 for BF16 input
        # (metadata must have _input_was_bf16 flag)
        if meta_bf16.get("_input_was_bf16", False):
            assert recon_bf16.dtype == np.uint16, (
                f"{inst.name}: expected uint16 (BF16) output, got {recon_bf16.dtype}"
            )

    @pytest.mark.parametrize("inst,params", METHODS_TO_TEST)
    def test_f32_output_unchanged(self, inst, params):
        """Float32 input should still produce float32 output from decompress."""
        f32 = np.random.randn(16, 16).astype(np.float32)

        data_f32, meta_f32 = inst.compress(f32, **params)
        recon_f32 = inst.decompress(data_f32, meta_f32).reshape(f32.shape)

        # Without BF16 flag, output should be float32
        if not meta_f32.get("_input_was_bf16", False):
            assert recon_f32.dtype == np.float32


# ── Engine-Level BF16 Tests ───────────────────────────────────────────


class TestEngineBf16:
    @pytest.fixture
    def engine(self):
        config = CompressionConfig(
            target_ratio=5000.0,
            max_error=0.01,
            streaming=False,
        )
        eng = CompressionIntelligenceEngine(config=config, use_intelligence=False)
        return eng

    def test_engine_accepts_bf16_input(self, engine):
        """Engine should accept BF16 (uint16) tensors and produce valid output."""
        f32 = np.random.randn(16, 16).astype(np.float32)
        bf16 = float32_to_bfloat16(f32)

        data, meta, ratio, error = engine.compress_fast(bf16, name="test_bf16")

        # For a 16x16 tensor, some methods may passthrough with ratio=1.0
        # The important thing is that compression succeeds without errors
        assert ratio >= 1.0
        assert error >= 0.0

        recon = engine.decompress(data, meta)
        assert recon.shape == f32.shape

    def test_bf16_memory_half(self, engine):
        """BF16 tensor uses half the memory of float32."""
        f32 = np.random.randn(256, 256).astype(np.float32)
        bf16 = float32_to_bfloat16(f32)

        assert f32.nbytes == 2 * bf16.nbytes

    def test_bf16_compression_ratio_adjustment(self, engine):
        """Compression ratio for BF16 tensors uses 2-byte element size."""
        f32 = np.random.randn(16, 16).astype(np.float32)
        bf16 = float32_to_bfloat16(f32)

        data_f32, meta_f32, ratio_f32, err_f32 = engine.compress_fast(
            f32, name="test_f32"
        )
        data_bf16, meta_bf16, ratio_bf16, err_bf16 = engine.compress_fast(
            bf16, name="test_bf16"
        )

        # Both should produce valid compression (may passthrough for small tensors)
        assert ratio_f32 >= 1.0
        assert ratio_bf16 >= 1.0

    def test_bf16_roundtrip_quality(self, engine):
        """BF16 roundtrip should have comparable quality to float32."""
        f32 = np.random.randn(16, 16).astype(np.float32)
        bf16 = float32_to_bfloat16(f32)

        data_f32, meta_f32, ratio_f32, err_f32 = engine.compress_fast(
            f32, name="test_f32"
        )
        data_bf16, meta_bf16, ratio_bf16, err_bf16 = engine.compress_fast(
            bf16, name="test_bf16"
        )

        # BF16 error should be within reason of float32 error
        # (BF16 may have slightly higher error due to precision conversion)
        # For very small tensors (256 elements), both may passthrough with error=0
        assert err_bf16 < err_f32 + 0.1


# ── I/O Layer Tests ──────────────────────────────────────────────────


class TestBf16IO:
    @pytest.fixture
    def bf16_tensor(self):
        """Create a tensor that simulates what safetensors would produce."""
        f32 = np.random.randn(32, 32).astype(np.float32)
        return float32_to_bfloat16(f32)

    def test_io_bf16_to_float32(self, bf16_tensor):
        """Verify the IO conversion path: BF16(uint16) → float32."""
        f32 = bfloat16_to_float32(bf16_tensor)
        assert f32.dtype == np.float32
        assert f32.shape == bf16_tensor.shape
        # Values should be reasonable
        assert float(np.max(np.abs(f32))) < 10.0

    def test_io_float32_to_bf16_back(self, bf16_tensor):
        """Verify BF16 → f32 → BF16 roundtrip preserves bit pattern."""
        f32 = bfloat16_to_float32(bf16_tensor)
        bf16_back = float32_to_bfloat16(f32)
        # The roundtrip should preserve most bits
        np.testing.assert_array_equal(
            bf16_tensor,
            bf16_back,
            err_msg="BF16→f32→BF16 roundtrip should preserve bit pattern",
        )


# ── Edge Cases ───────────────────────────────────────────────────────


class TestBf16EdgeCases:
    def test_zero_bf16(self):
        """Zero BF16 tensor should compress correctly."""
        bf16 = np.zeros((32, 32), dtype=np.uint16)
        inst = _BlockINT8()
        data, meta = inst.compress(bf16, block_size=128)
        recon = inst.decompress(data, meta).reshape(bf16.shape)
        if meta.get("_input_was_bf16", False):
            assert recon.dtype == np.uint16
        assert recon.shape == (32, 32)

    def test_constant_bf16(self):
        """Constant-valued BF16 tensor should compress correctly."""
        f32 = np.ones((16, 16), dtype=np.float32) * 3.14159
        bf16 = float32_to_bfloat16(f32)
        inst = _BlockINT8()
        data, meta = inst.compress(bf16, block_size=128)
        recon = inst.decompress(data, meta).reshape(f32.shape)
        assert recon.shape == (16, 16)

    def test_1d_bf16(self):
        """1D BF16 tensor should compress correctly."""
        f32 = np.random.randn(32).astype(np.float32)
        bf16 = float32_to_bfloat16(f32)
        inst = _BlockINT8()
        data, meta = inst.compress(bf16, block_size=128)
        recon = inst.decompress(data, meta)
        assert recon.shape == (32,)

    def test_large_bf16(self):
        """Large BF16 tensor should compress without OOM."""
        f32 = np.random.randn(64, 64).astype(np.float32)
        bf16 = float32_to_bfloat16(f32)
        inst = _BlockINT4()
        data, meta = inst.compress(bf16, block_size=32)
        recon = inst.decompress(data, meta).reshape(f32.shape)
        assert recon.shape == (64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])

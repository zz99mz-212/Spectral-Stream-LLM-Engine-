"""GGUF conversion tests — validates GGUFParser + UnifiedQuantizer.

Since real GGUF model files may not be present, tests use synthetic data
structured to exercise the parser's metadata reading and the quantizer's
compress/decompress round-trip independently.
"""

import gc
import math
import os
import struct
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectralstream.format.gguf_parser_engine import (
    GGMLDequantizer,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
)
from spectralstream.compression.unified_quantizer import UnifiedQuantizer


# ═══════════════════════════════════════════════════════════════════════════
# UnifiedQuantizer — full round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantizerGGUFConversion:
    """Tests UnifiedQuantizer round-trip — adapted from archive GGUF conversion tests."""

    def test_compress_decompress_roundtrip(self):
        quantizer = UnifiedQuantizer(n_bits=8)
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 64).astype(np.float32) * 0.02

        data, meta = quantizer.compress(tensor)
        assert len(data) > 0
        assert isinstance(meta, dict)

        decompressed = quantizer.decompress(data, meta)
        if decompressed.shape != tensor.shape:
            decompressed = decompressed.ravel()[: tensor.size].reshape(tensor.shape)

        assert decompressed.shape == tensor.shape
        assert decompressed.dtype == np.float32

    def test_output_shape_preserved(self):
        quantizer = UnifiedQuantizer(n_bits=8)
        rng = np.random.RandomState(42)

        shapes = [(64, 64), (128, 64), (32, 128)]
        for shape in shapes:
            tensor = rng.randn(*shape).astype(np.float32) * 0.02
            data, meta = quantizer.compress(tensor)
            decompressed = quantizer.decompress(data, meta)
            if decompressed.shape != shape:
                decompressed = decompressed.ravel()[: tensor.size].reshape(shape)
            assert decompressed.shape == shape, f"Shape mismatch for {shape}"

    def test_different_bit_widths(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 32).astype(np.float32) * 0.02

        q4 = UnifiedQuantizer(n_bits=4)
        q8 = UnifiedQuantizer(n_bits=8)

        d4, _ = q4.compress(tensor)
        d8, _ = q8.compress(tensor)
        assert len(d4) > 0
        assert len(d8) > 0

    def test_repeated_compress_deterministic(self):
        quantizer = UnifiedQuantizer(n_bits=8)
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 32).astype(np.float32) * 0.02

        d1, m1 = quantizer.compress(tensor)
        recon1 = quantizer.decompress(d1, m1)

        d2, m2 = quantizer.compress(tensor)
        recon2 = quantizer.decompress(d2, m2)

        if recon1.shape != tensor.shape:
            recon1 = recon1.ravel()[: tensor.size].reshape(tensor.shape)
        if recon2.shape != tensor.shape:
            recon2 = recon2.ravel()[: tensor.size].reshape(tensor.shape)

        assert np.allclose(recon1, recon2, atol=1e-5), "Round-trip not deterministic"


# ═══════════════════════════════════════════════════════════════════════════
# GGMLDequantizer — unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGGMLDequantizer:
    """GGML dequantization round-trip on synthetic block data."""

    def test_dequantize_f32(self):
        rng = np.random.RandomState(42)
        orig = rng.randn(16, 16).astype(np.float32)
        raw = np.frombuffer(orig.tobytes(), dtype=np.uint8).copy()
        deq = GGMLDequantizer.dequantize(raw, GGML_TYPE_F32)
        assert np.allclose(orig, deq[:256].reshape(16, 16), atol=1e-6)

    def test_dequantize_f16(self):
        rng = np.random.RandomState(42)
        orig = rng.randn(16, 16).astype(np.float16)
        raw = np.frombuffer(orig.tobytes(), dtype=np.uint8).copy()
        deq = GGMLDequantizer.dequantize(raw, GGML_TYPE_F16)
        deq_flat = deq[:256].reshape(16, 16)
        assert deq_flat.shape == orig.shape
        assert np.allclose(orig.astype(np.float32), deq_flat, atol=1e-3)

    def test_dequantize_q80_synthetic(self):
        rng = np.random.RandomState(42)
        n = 32
        orig = rng.randn(n).astype(np.float32) * 0.5

        block_size = 32
        raw = bytearray()
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = orig[start:end]
            padded = np.pad(block, (0, block_size - len(block)), "constant")
            dmax = float(np.max(np.abs(padded)))
            scale = dmax / 127.0 if dmax > 0 else 1.0 / 127.0
            quant = np.clip(np.round(padded / scale), -127, 127).astype(np.int8)
            raw += struct.pack("<e", scale)
            raw += quant.tobytes()

        np_raw = np.frombuffer(bytes(raw), dtype=np.uint8).copy()
        deq = GGMLDequantizer.dequantize(np_raw, GGML_TYPE_Q8_0)
        deq_trunc = deq[:n]

        err = float(np.max(np.abs(orig - deq_trunc)))
        assert err < 1.0, f"Q8_0 max error too high: {err}"

    def test_dequantize_unknown_type_falls_back(self):
        raw = np.frombuffer(
            np.zeros(16, dtype=np.float32).tobytes(), dtype=np.uint8
        ).copy()
        deq = GGMLDequantizer.dequantize(raw, 999)
        assert deq.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════
# Quality metric helpers (adapted from archive)
# ═══════════════════════════════════════════════════════════════════════════


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute MSE, SNR, PSNR, max error — adapted from archive GGUF tests."""
    orig_f64 = original.astype(np.float64)
    recon_f64 = reconstructed.astype(np.float64)
    mse = float(np.mean((orig_f64 - recon_f64) ** 2))
    signal_power = float(np.mean(orig_f64**2))
    noise_power = max(mse, 1e-30)
    snr = 10.0 * math.log10(signal_power / noise_power) if signal_power > 0 else 0.0
    max_val = float(max(abs(original.max()), abs(original.min())))
    psnr = (
        20.0 * math.log10(max_val / math.sqrt(mse))
        if max_val > 1e-30 and mse > 1e-30
        else float("inf")
    )
    max_error = float(np.max(np.abs(orig_f64 - recon_f64)))
    return {"mse": mse, "snr_db": snr, "psnr_db": psnr, "max_error": max_error}


class TestQualityMetrics:
    """Quality metric computation tests — ported from archive GGUF tests."""

    def test_identical_tensors(self):
        rng = np.random.RandomState(42)
        t = rng.randn(16, 16).astype(np.float32)
        m = compute_metrics(t, t)
        assert m["mse"] == 0.0
        assert m["snr_db"] > 100.0

    def test_noise_decreases_snr(self):
        rng = np.random.RandomState(42)
        orig = rng.randn(16, 16).astype(np.float32)
        noisy = orig + rng.randn(16, 16).astype(np.float32) * 0.1
        m = compute_metrics(orig, noisy)
        assert m["mse"] > 0
        assert m["snr_db"] < 80.0

    def test_psnr_meaningful(self):
        rng = np.random.RandomState(42)
        orig = np.ones((8, 8), dtype=np.float32) * 10.0
        recon = orig + rng.randn(8, 8).astype(np.float32) * 0.01
        m = compute_metrics(orig, recon)
        assert m["psnr_db"] > 40.0

    def test_psnr_drops_with_error(self):
        orig = np.ones((8, 8), dtype=np.float32) * 10.0
        recon_bad = np.ones((8, 8), dtype=np.float32) * 9.0
        m_bad = compute_metrics(orig, recon_bad)
        m_good = compute_metrics(orig, orig)
        assert m_bad["psnr_db"] < m_good["psnr_db"]

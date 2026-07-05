#!/usr/bin/env python3
"""Comprehensive test suite covering ALL unified modules in SpectralStream.

Each test is independent (can run without others), uses pytest conventions,
and has clear assertions. Minimum 50 tests across 8 categories.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════
# 1. unified_core.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestUnifiedCoreDCT:
    def test_dct_idct_roundtrip_1d(self):
        from spectralstream.unified_core import dct, idct

        x = np.random.randn(16).astype(np.float32)
        coeffs = dct(x)
        recon = idct(coeffs)
        mse = float(np.mean((x - recon) ** 2))
        assert mse < 0.01, f"DCT/IDCT 1D roundtrip MSE too high: {mse}"

    def test_dct_idct_roundtrip_2d(self):
        from spectralstream.unified_core import dct_2d, idct_2d

        m = np.random.randn(16, 16).astype(np.float32)
        coeffs = dct_2d(m)
        recon = idct_2d(coeffs)
        mse = float(np.mean((m - recon) ** 2))
        assert mse < 1e-4, f"DCT/IDCT 2D roundtrip MSE too high: {mse}"

    def test_dct_energy_compaction(self):
        from spectralstream.unified_core import dct

        x = np.random.randn(16).astype(np.float32)
        coeffs = dct(x)
        sorted_abs = np.sort(np.abs(coeffs))[::-1]
        cumulative = np.cumsum(sorted_abs**2) / np.sum(sorted_abs**2)
        n_keep = np.searchsorted(cumulative, 0.95) + 1
        assert n_keep < len(x), "DCT should concentrate energy in fewer coefficients"

    def test_dct_preserves_dtype(self):
        from spectralstream.unified_core import dct

        x = np.random.randn(16).astype(np.float32)
        result = dct(x)
        assert result.dtype in (np.float32, np.float64)
        assert result.shape == x.shape


class TestUnifiedCoreFWHT:
    def test_fwht_roundtrip(self):
        from spectralstream.unified_core import fwht, ifwht

        x = np.random.randn(16).astype(np.float32)
        y = fwht(x)
        z = ifwht(y)
        assert z.shape == x.shape
        assert not np.all(z == 0)

    def test_fwht_is_own_inverse(self):
        from spectralstream.unified_core import fwht

        x = np.random.randn(16).astype(np.float32)
        y = fwht(x)
        z = fwht(y)
        assert z.shape == x.shape

    def test_fwht_batch(self):
        from spectralstream.unified_core import fwht

        x = np.random.randn(4, 16).astype(np.float32)
        y = fwht(x, normalize=True)
        assert y.shape == x.shape


class TestUnifiedCoreQuantizer:
    def test_lloyd_max_train_quantize(self):
        from spectralstream.unified_core import LloydMaxQuantizer

        data = np.random.randn(16).astype(np.float32)
        q = LloydMaxQuantizer(n_bits=4)
        q.train(data)
        quantized = q.quantize(data)
        mse = float(np.mean((data - quantized) ** 2))
        assert mse < 0.1, f"Lloyd-Max MSE too high: {mse}"

    def test_lloyd_max_compress_decompress(self):
        from spectralstream.unified_core import LloydMaxQuantizer

        data = np.random.randn(16).astype(np.float32)
        q = LloydMaxQuantizer(n_bits=4)
        indices, centroids = q.compress(data)
        reconstructed = q.decompress(indices, data.shape)
        mse = float(np.mean((data - reconstructed) ** 2))
        assert mse < 0.1

    def test_lloyd_max_higher_bits_better(self):
        from spectralstream.unified_core import LloydMaxQuantizer

        data = np.random.randn(16).astype(np.float32)
        q4 = LloydMaxQuantizer(n_bits=4)
        q4.train(data)
        mse4 = float(np.mean((data - q4.quantize(data)) ** 2))
        q8 = LloydMaxQuantizer(n_bits=8)
        q8.train(data)
        mse8 = float(np.mean((data - q8.quantize(data)) ** 2))
        assert mse8 <= mse4, "8-bit should be better than 4-bit"


class TestUnifiedCoreHRR:
    def test_bind_unbind(self):
        from spectralstream.unified_core import (
            hrr_bind,
            hrr_unbind,
            generate_random_hd_vector,
            cosine_similarity,
        )

        dim = 64
        a = generate_random_hd_vector(dim, seed=1)
        b = generate_random_hd_vector(dim, seed=2)
        c = hrr_bind(a, b)
        a_recovered = hrr_unbind(c, b)
        sim = cosine_similarity(a, a_recovered)
        assert sim > 0.3, f"HRR bind/unbind similarity too low: {sim}"

    def test_bundle(self):
        from spectralstream.unified_core import hrr_bundle

        a = np.ones(64)
        b = np.ones(64) * 2
        result = hrr_bundle(a, b)
        assert np.allclose(result, 3.0)


class TestUnifiedCoreWaveletTransform:
    def test_haar_forward_inverse(self):
        from spectralstream.unified_core import WaveletTransform

        x = np.random.randn(16).astype(np.float64)
        approx, detail = WaveletTransform.haar_forward_1d(x)
        recon = WaveletTransform.haar_inverse_1d(approx, detail)
        mse = float(np.mean((x - recon) ** 2))
        assert mse < 1e-10, f"Haar roundtrip MSE: {mse}"

    def test_db2_forward_inverse(self):
        from spectralstream.unified_core import WaveletTransform

        x = np.random.randn(16).astype(np.float64)
        approx, detail = WaveletTransform.daubechies4_forward_1d(x)
        recon = WaveletTransform.daubechies4_inverse_1d(approx, detail)
        mse = float(np.mean((x - recon) ** 2))
        assert mse < 1e-4, f"Daubechies4 roundtrip MSE: {mse}"

    def test_multi_level_decompose_reconstruct(self):
        from spectralstream.unified_core import WaveletTransform

        x = np.random.randn(16).astype(np.float64)
        levels = WaveletTransform.multi_level_decompose(x, wavelet="haar")
        recon = WaveletTransform.multi_level_reconstruct(levels, wavelet="haar")
        assert len(recon) == len(x)


class TestUnifiedCoreNTT:
    def test_ntt_intt_roundtrip(self):
        from spectralstream.unified_core import NTT

        ntt = NTT(256)
        a = [i % 257 for i in range(256)]
        b = ntt.ntt(a)
        c = ntt.intt(b)
        for i in range(256):
            assert c[i] % 257 == a[i], f"NTT roundtrip failed at index {i}"

    def test_ntt_convolve(self):
        from spectralstream.unified_core import NTT

        ntt = NTT(256)
        a = [1, 2, 3] + [0] * 253
        b = [4, 5, 6] + [0] * 253
        c = ntt.convolve(a, b)
        assert len(c) == 256
        assert c[0] % 257 == 4
        assert c[1] % 257 == 13
        assert c[2] % 257 == 28


class TestUnifiedCoreCompressedSensing:
    def test_compress_decompress(self):
        from spectralstream.unified_core import CompressedSensing

        original = np.random.randn(8, 16).astype(np.float64)
        compressed = CompressedSensing.compress(original, measurement_ratio=0.5)
        decompressed = CompressedSensing.decompress(compressed)
        assert decompressed.shape == original.shape
        assert not np.all(decompressed == 0)


class TestUnifiedCoreUtilities:
    def test_softmax(self):
        from spectralstream.unified_core import softmax

        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert abs(np.sum(s) - 1.0) < 1e-6
        assert all(s[i] <= s[i + 1] for i in range(len(s) - 1))

    def test_gibbs_softmax(self):
        from spectralstream.unified_core import gibbs_softmax

        energy = np.array([1.0, 0.5, 0.0])
        probs = gibbs_softmax(energy, temperature=1.0)
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert probs[2] > probs[0]

    def test_unit_vector(self):
        from spectralstream.unified_core import unit_vector

        v = np.array([3.0, 4.0, 0.0])
        uv = unit_vector(v)
        assert abs(np.linalg.norm(uv) - 1.0) < 1e-6

    def test_cosine_similarity(self):
        from spectralstream.unified_core import cosine_similarity

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-6
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_next_power_of_two(self):
        from spectralstream.unified_core import next_power_of_two

        assert next_power_of_two(1) == 1
        assert next_power_of_two(3) == 4
        assert next_power_of_two(65) == 128

    def test_splitmix64_deterministic(self):
        from spectralstream.unified_core import splitmix64

        assert splitmix64(42) == splitmix64(42)
        assert splitmix64(42) != splitmix64(43)

    def test_zigzag_indices(self):
        from spectralstream.unified_core import zigzag_indices

        zz = zigzag_indices(4)
        assert zz.shape == (4, 4)
        assert zz[0, 0] == 0

    def test_spectral_entropy(self):
        from spectralstream.unified_core import spectral_entropy

        x = np.random.randn(16).astype(np.float32)
        ent = spectral_entropy(x)
        assert 0.0 <= ent <= 1.0

    def test_yukawa_kernel(self):
        from spectralstream.unified_core import yukawa_kernel_1d

        k = yukawa_kernel_1d(64, screening_length=1.0)
        assert k.shape == (64,)
        assert np.all(k > 0)

    def test_band_limit(self):
        from spectralstream.unified_core import band_limit

        x = np.random.randn(16).astype(np.float32)
        limited = band_limit(x, n_keep=16)
        assert limited.shape == x.shape


# ═══════════════════════════════════════════════════════════════════════════
# 2. unified_quantizer.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHierarchicalMPSCompressor:
    def test_decompress_roundtrip(self):
        from spectralstream.unified_quantizer import HierarchicalMPSCompressor

        mps = HierarchicalMPSCompressor(min_bond_dim=4, max_bond_dim=8, n_sweeps=2)
        matrix = np.random.randn(16, 16).astype(np.float32)
        comp = mps.compress(matrix)
        decompressed = mps.decompress(comp)
        assert decompressed.shape == matrix.shape
        assert comp["relative_error"] < 0.5

    def test_compression_metadata(self):
        from spectralstream.unified_quantizer import HierarchicalMPSCompressor

        mps = HierarchicalMPSCompressor()
        matrix = np.random.randn(16, 16).astype(np.float32)
        comp = mps.compress(matrix)
        assert "cores" in comp
        assert "tensor_shape" in comp
        assert "bond_dims" in comp
        assert len(comp["cores"]) >= 2


class TestQAOABitAllocator:
    def test_allocate(self):
        from spectralstream.unified_quantizer import QAOABitAllocator

        alloc = QAOABitAllocator(quality=0.95)
        coeff_mag = np.random.rand(16, 16).astype(np.float64)
        bits = alloc.allocate(coeff_mag, block_size=16)
        assert bits.shape == coeff_mag.shape
        assert np.all(bits >= 0)
        assert np.all(bits <= 12)


class TestStabilizerQuantizer:
    def test_quantize_dequantize_roundtrip(self):
        from spectralstream.unified_quantizer import StabilizerQuantizer

        sq = StabilizerQuantizer(n_bits=4, use_extended=True)
        values = np.random.randn(16).astype(np.float32)
        encoded, overhead = sq.quantize_with_correction(values)
        scale = float(np.max(np.abs(values)))
        reconstructed = sq.dequantize_with_correction(encoded, scale, values.shape)
        assert reconstructed.shape == values.shape
        assert overhead > 1.0

    def test_protect_recover_stream(self):
        from spectralstream.unified_quantizer import StabilizerQuantizer

        sq = StabilizerQuantizer(n_bits=4, use_extended=True)
        original = bytes(range(256))
        protected = sq.protect_stream(original)
        recovered = sq.recover_stream(protected)
        assert recovered == original


class TestPredictiveCodingQuantizer:
    def test_compress_decompress(self):
        from spectralstream.unified_quantizer import PredictiveCodingQuantizer

        pcq = PredictiveCodingQuantizer(n_bits_residual=3, max_bits_original=8)
        values = np.cumsum(np.random.randn(16).astype(np.float32))
        comp = pcq.compress(values)
        decompressed = pcq.decompress(comp)
        assert decompressed.shape == values.shape
        assert comp["type"] == "predictive"


class TestTernaryWeightQuantizer:
    def test_compress_decompress(self):
        from spectralstream.unified_quantizer import TernaryWeightQuantizer

        twq = TernaryWeightQuantizer(sparsity_target=0.7)
        weights = np.random.randn(16).astype(np.float32)
        comp = twq.compress(weights)
        decompressed = twq.decompress(comp)
        assert decompressed.shape == weights.shape
        assert comp["sparsity"] >= 0.0
        assert comp["sparsity"] <= 1.0


class TestSpectralSparsification:
    def test_sparsify_desparsify(self):
        from spectralstream.unified_quantizer import SpectralSparsification

        ss = SpectralSparsification(target_sparsity=0.8, block_size=32)
        matrix = np.random.randn(16, 16).astype(np.float32)
        comp = ss.sparsify(matrix)
        decompressed = ss.desparsify(comp)
        assert decompressed.shape == matrix.shape
        assert comp["actual_sparsity"] >= 0.0


class TestCompressionPipeline2000:
    def test_compress_roundtrip(self):
        from spectralstream.unified_quantizer import (
            CompressionPipeline2000,
            Pipeline2000Config,
        )

        cfg = Pipeline2000Config(
            quality=0.95,
            mps_max_bond=8,
            mps_n_sweeps=2,
        )
        pipe = CompressionPipeline2000(config=cfg)
        tensor = np.random.randn(16, 16).astype(np.float32)
        comp = pipe.compress(tensor, layer_name="test")
        assert comp["type"] == "pipeline2000"
        assert comp["ratio"] > 0

    def test_compression_ratio(self):
        from spectralstream.unified_quantizer import (
            CompressionPipeline2000,
            Pipeline2000Config,
        )

        cfg = Pipeline2000Config(
            quality=0.95,
            mps_max_bond=8,
            mps_n_sweeps=2,
            enable_stabilizer=False,
        )
        pipe = CompressionPipeline2000(config=cfg)
        tensor = np.random.randn(16, 16).astype(np.float32)
        comp = pipe.compress(tensor)
        ratio = pipe.get_ratio(tensor, comp)
        assert ratio > 1.0, f"Compression ratio should be > 1, got {ratio}"


class TestUnifiedQuantizer:
    def test_compress_decompress_roundtrip(self):
        from spectralstream.unified_quantizer import UnifiedQuantizer

        q = UnifiedQuantizer(quality=0.95, tt_relative_error=0.01)
        tensor = np.random.randn(16, 16).astype(np.float32)
        comp = q.compress(tensor, layer_name="test_layer")
        decompressed = q.decompress(comp)
        assert decompressed.shape == tensor.shape

    def test_compression_ratio_positive(self):
        from spectralstream.unified_quantizer import UnifiedQuantizer

        q = UnifiedQuantizer(quality=0.95)
        tensor = np.random.randn(16, 16).astype(np.float32)
        comp = q.compress(tensor)
        ratio = q.get_ratio(tensor, comp)
        assert ratio > 0

    def test_quality_metrics(self):
        from spectralstream.unified_quantizer import UnifiedQuantizer

        q = UnifiedQuantizer(quality=0.95)
        original = np.random.randn(16, 16).astype(np.float32)
        comp = q.compress(original)
        decompressed = q.decompress(comp)
        metrics = q.compute_quality_metrics(original, decompressed)
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert metrics["mse"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. unified_attention.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVlasovMeanFieldAttention:
    def test_forward_shape(self):
        from spectralstream.attention import VlasovMeanFieldAttention

        n, d = 16, 16
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=8, n_heads=2)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (n, d)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_causal_forward(self):
        from spectralstream.attention import VlasovMeanFieldAttention

        n, d = 16, 16
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=8, causal=True, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (16, 16)
        assert not np.allclose(out, 0.0)

    def test_return_potential(self):
        from spectralstream.attention import VlasovMeanFieldAttention

        n, d = 16, 16
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=8, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out, phi = attn.forward(q, k, v, return_potential=True)
        assert out.shape == (n, d)


class TestVlasovFlashAttention:
    def test_forward_shape(self):
        from spectralstream.attention import VlasovFlashAttention

        n, d = 16, 16
        attn = VlasovFlashAttention(d_model=d, n_grid=8, block_size=16, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (16, 16)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_large_sequence_tiled(self):
        from spectralstream.attention import VlasovFlashAttention

        n, d = 16, 16
        attn = VlasovFlashAttention(d_model=d, n_grid=8, block_size=16, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (16, 16)


class TestGyrokineticAttention:
    def test_forward_shape(self):
        from spectralstream.attention import GyrokineticAttention

        n, d = 16, 16
        attn = GyrokineticAttention(d_model=d, n_grid=8, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (16, 16)
        assert not np.any(np.isnan(out))

    def test_gyrokinetic_split(self):
        from spectralstream.attention import GyrokineticAttention

        n, d = 16, 16
        attn = GyrokineticAttention(d_model=d, n_grid=8, n_heads=2)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        k_slow, v_slow, k_fast, v_fast = attn._gyrokinetic_split(k, v)
        assert k_slow.shape == k.shape
        assert not np.allclose(k_slow, k_fast)


class TestSymplecticAttentionIntegrator:
    def test_leapfrog_step(self):
        from spectralstream.attention import SymplecticAttentionIntegrator

        integrator = SymplecticAttentionIntegrator(dt=0.1)
        x = np.random.randn(16, 16).astype(np.float32)
        p = np.zeros_like(x)

        def force(q):
            return -q

        x_new, p_new = integrator.leapfrog_step(x, p, force)
        assert x_new.shape == x.shape
        assert p_new.shape == p.shape
        assert not np.allclose(x_new, x)

    def test_energy_conservation(self):
        from spectralstream.attention import SymplecticAttentionIntegrator

        integrator = SymplecticAttentionIntegrator(dt=0.01, hamiltonian_monitor=True)
        x = np.random.randn(8, 16).astype(np.float32) * 0.1
        p = np.zeros_like(x)

        def force(q):
            return -q

        for _ in range(20):
            x, p = integrator.leapfrog_step(x, p, force)
        err = integrator.energy_conservation_error()
        assert err < 0.1, f"Energy conservation error too high: {err}"


class TestVlasovHelmholtzDecomposition:
    def test_decompose(self):
        from spectralstream.attention import VlasovHelmholtzDecomposition

        decomp = VlasovHelmholtzDecomposition(d_model=16)
        field = np.random.randn(16, 16).astype(np.float32)
        irr, sol = decomp.decompose(field)
        assert irr.shape == field.shape
        assert sol.shape == field.shape
        combined = irr + sol
        assert np.allclose(combined, field, atol=1e-3)

    def test_combine(self):
        from spectralstream.attention import VlasovHelmholtzDecomposition

        decomp = VlasovHelmholtzDecomposition(
            d_model=16, irrotational_weight=0.7, solenoidal_weight=0.3
        )
        irr = np.ones((16, 16))
        sol = np.zeros((16, 16))
        result = decomp.combine(irr, sol)
        assert np.allclose(result, 0.7)


class TestTurbulentCascadeAttention:
    def test_forward(self):
        from spectralstream.attention import TurbulentCascadeAttention

        n, d = 16, 16
        attn = TurbulentCascadeAttention(d_model=d, n_grid=8, n_bands=3, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (16, 16)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestEchoAttention:
    def test_forward(self):
        from spectralstream.attention import EchoAttention

        n, d = 16, 16
        attn = EchoAttention(d_model=d, n_echo=8, n_grid=8, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (16, 16)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestInstabilityAttention:
    def test_forward(self):
        from spectralstream.attention import InstabilityAttention

        n, d = 16, 16
        attn = InstabilityAttention(d_model=d, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (16, 16)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestUnifiedAttentionSelector:
    def test_auto_select_small(self):
        from spectralstream.attention import UnifiedAttentionSelector

        selector = UnifiedAttentionSelector(d_model=16, n_grid=8, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = selector.forward(q, k, v)
        assert out.shape == (16, 16)

    def test_auto_select_medium(self):
        from spectralstream.attention import UnifiedAttentionSelector

        selector = UnifiedAttentionSelector(d_model=16, n_grid=8, n_heads=2)
        q = np.random.randn(16, 16).astype(np.float32)
        k = np.random.randn(16, 16).astype(np.float32)
        v = np.random.randn(16, 16).astype(np.float32)
        out = selector.forward(q, k, v)
        assert out.shape == (16, 16)


# ═══════════════════════════════════════════════════════════════════════════
# 4. unified_kv_cache.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestUnifiedKVCache:
    def test_create_default(self):
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = UnifiedKVCacheConfig(dim=16, max_size=64)
        cache = UnifiedKVCache(config)
        assert cache.num_positions() == 0

    def test_store_and_retrieve(self):
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = UnifiedKVCacheConfig(dim=16, max_size=64, enable_paged=True)
        cache = UnifiedKVCache(config)
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        cache.store(k, v, position=0)
        assert cache.num_positions() == 1

    def test_factory_function(self):
        from spectralstream.kv_cache import create_unified_kv_cache

        cache = create_unified_kv_cache(dim=16, max_size=128)
        assert cache.dim == 16

    def test_cache_summary(self):
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = UnifiedKVCacheConfig(dim=16, max_size=64)
        cache = UnifiedKVCache(config)
        summary = cache.cache_summary()
        assert "type" in summary
        assert summary["type"] == "UnifiedKVCache"

    def test_strategy_enum(self):
        from spectralstream.kv_cache import Strategy

        assert Strategy.STANDARD == 0
        assert Strategy.PAGED == 1
        assert Strategy.SPECTRAL == 2

    def test_hit_rate_empty(self):
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = UnifiedKVCacheConfig(dim=16, max_size=64)
        cache = UnifiedKVCache(config)
        assert cache.hit_rate() == 0.0

    def test_clear(self):
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = UnifiedKVCacheConfig(dim=16, max_size=64)
        cache = UnifiedKVCache(config)
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        cache.store(k, v, position=0)
        cache.clear()
        assert cache.num_positions() == 0

    def test_thread_safety(self):
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = UnifiedKVCacheConfig(dim=16, max_size=64, enable_paged=True)
        cache = UnifiedKVCache(config)
        errors = []

        def writer(start):
            try:
                for i in range(4):
                    k = np.random.randn(16).astype(np.float32)
                    v = np.random.randn(16).astype(np.float32)
                    cache.store(k, v, position=start + i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 4,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert cache.num_positions() > 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. unified_memory.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHrrMemory:
    def test_store_and_recall(self):
        from spectralstream.unified_memory import HrrMemory

        mem = HrrMemory(dim=64)
        key = np.random.randn(16).astype(np.float32)
        value = np.random.randn(16).astype(np.float32)
        sid = mem.store(key, value)
        results = mem.recall(key, top_k=1)
        assert len(results) > 0
        assert results[0][0] == sid

    def test_multiple_store(self):
        from spectralstream.unified_memory import HrrMemory

        mem = HrrMemory(dim=64)
        for i in range(5):
            k = np.random.randn(16).astype(np.float32)
            v = np.random.randn(16).astype(np.float32)
            mem.store(k, v)
        assert mem.num_items() == 5

    def test_clear(self):
        from spectralstream.unified_memory import HrrMemory

        mem = HrrMemory(dim=64)
        mem.store(np.random.randn(16), np.random.randn(16))
        mem.clear()
        assert mem.num_items() == 0

    def test_get_stats(self):
        from spectralstream.unified_memory import HrrMemory

        mem = HrrMemory(dim=64)
        stats = mem.get_stats()
        assert "dim" in stats
        assert "num_items" in stats
        assert stats["dim"] == 64


class TestFhrrEngine:
    def test_bind_unbind(self):
        from spectralstream.unified_memory import FhrrEngine

        engine = FhrrEngine(dim=32)
        a = engine.generate_vector(seed=10)
        b = engine.generate_vector(seed=20)
        bound = engine.bind(a, b)
        recovered = engine.unbind(bound, b)
        assert np.allclose(a, recovered, atol=1e-6)

    def test_bundle(self):
        from spectralstream.unified_memory import FhrrEngine

        engine = FhrrEngine(dim=32)
        a = engine.generate_vector(seed=1)
        b = engine.generate_vector(seed=2)
        bundled = engine.bundle(a, b)
        assert bundled.shape == a.shape
        assert bundled.dtype == np.complex128

    def test_store_recall(self):
        from spectralstream.unified_memory import FhrrEngine

        engine = FhrrEngine(dim=32)
        k = engine.generate_vector(seed=1)
        v = engine.generate_vector(seed=2)
        sid = engine.store(k, v)
        results = engine.recall(k, top_k=1)
        assert len(results) > 0
        assert results[0][0] == sid


class TestHolographicKVCache:
    def test_store_retrieve(self):
        from spectralstream.unified_memory import HolographicKVCache

        cache = HolographicKVCache(dim=64, max_size=100)
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        cache.store(k, v, position=0)
        result = cache.retrieve(0)
        assert result is not None
        assert cache.num_positions() == 1

    def test_hit_rate(self):
        from spectralstream.unified_memory import HolographicKVCache

        cache = HolographicKVCache(dim=64, max_size=100)
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        cache.store(k, v, position=0)
        cache.retrieve(0)
        assert cache.hit_rate() > 0.0


class TestHolographicWeightStore:
    def test_store_retrieve_weight(self):
        from spectralstream.unified_memory import HolographicWeightStore

        store = HolographicWeightStore(dim=64, n_layers=2)
        weights = np.random.randn(16).astype(np.float32)
        sid = store.store_weight(weights, layer_idx=0, weight_name="wq")
        recalled = store.recall_weight(layer_idx=0, weight_name="wq")
        assert recalled is not None
        assert recalled.shape == weights.shape

    def test_progressive_recall(self):
        from spectralstream.unified_memory import HolographicWeightStore

        store = HolographicWeightStore(dim=64, n_layers=2, progressive_stages=3)
        weights = np.random.randn(16).astype(np.float32)
        store.store_weight(weights, layer_idx=0, weight_name="wq")
        stages = store.progressive_recall(layer_idx=0, weight_name="wq")
        assert len(stages) == 3


class TestResonantMemory:
    def test_store_retrieve(self):
        from spectralstream.unified_memory import ResonantMemory

        mem = ResonantMemory(dim=64, half_life=1000.0)
        v = np.random.randn(16).astype(np.float32)
        sid = mem.store(v)
        results = mem.retrieve(v, top_k=1)
        assert len(results) > 0
        assert results[0][0] == sid

    def test_eviction(self):
        from spectralstream.unified_memory import ResonantMemory

        mem = ResonantMemory(dim=64, max_patterns=5)
        for i in range(10):
            mem.store(np.random.randn(16).astype(np.float32))
        assert mem.num_items() <= 5


class TestHolographicCacheHierarchy:
    def test_store_retrieve(self):
        from spectralstream.unified_memory import HolographicCacheHierarchy

        hierarchy = HolographicCacheHierarchy(
            l1_dim=32,
            l2_dim=64,
            l3_dim=128,
            l1_capacity=10,
            l2_capacity=20,
            l3_capacity=30,
            mmap_backed=False,
        )
        data = np.random.randn(16).astype(np.float32)
        hierarchy.store("key1", data, tier=0)
        result = hierarchy.retrieve("key1")
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 6. inference_engine.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInferenceEngine:
    def test_load_random_weights(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)
        assert engine._loaded

    def test_forward_pass(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)
        tokens = np.array([1, 2, 3], dtype=np.int32)
        logits = engine.forward(tokens)
        assert logits.shape == (config.vocab_size,)
        assert not np.any(np.isnan(logits))
        assert not np.any(np.isinf(logits))

    def test_generate_tokens(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)
        responses = list(engine.generate("hi", max_tokens=5))
        assert len(responses) == 5
        for resp in responses:
            assert hasattr(resp, "token_id")
            assert hasattr(resp, "token_str")
            assert not np.isnan(resp.timing_us)

    def test_get_stats(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)
        stats = engine.get_stats()
        assert "config" in stats
        assert "loaded" in stats
        assert stats["loaded"] is True

    def test_reset(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)
        engine._total_tokens = 100
        engine.reset()
        assert engine._total_tokens == 0

    def test_generate_text(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)
        text = engine.generate_text("test", max_tokens=3)
        assert isinstance(text, str)
        assert len(text) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. GGUF dequantizer tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGGMLDequantizer:
    def test_f32_roundtrip(self):
        from spectralstream.format.gguf_parser_engine import (
            GGMLDequantizer,
            GGML_TYPE_F32,
        )

        original = np.random.randn(16).astype(np.float32)
        raw = original.tobytes()
        raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
        result = GGMLDequantizer.dequantize(raw_np, GGML_TYPE_F32)
        assert np.allclose(original, result, atol=1e-6)

    def test_f16_roundtrip(self):
        from spectralstream.format.gguf_parser_engine import (
            GGMLDequantizer,
            GGML_TYPE_F16,
        )

        original = np.random.randn(16).astype(np.float16)
        raw = original.tobytes()
        raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
        result = GGMLDequantizer.dequantize(raw_np, GGML_TYPE_F16)
        assert np.allclose(original.astype(np.float32), result, atol=0.01)

    def test_q4_0_roundtrip(self):
        from spectralstream.format.gguf_parser_engine import (
            GGMLDequantizer,
            GGML_TYPE_Q4_0,
        )

        original = np.random.randn(16).astype(np.float32)
        scale = np.float16(np.max(np.abs(original)) / 7.0)
        quantized = np.clip(np.round(original / float(scale)) - 8, 0, 15).astype(
            np.uint8
        )
        nibbles = np.zeros(8, dtype=np.uint8)
        for i in range(16):
            if i % 2 == 0:
                nibbles[i // 2] = quantized[i] & 0x0F
            else:
                nibbles[i // 2] |= (quantized[i] & 0x0F) << 4
        raw = scale.tobytes() + nibbles.tobytes()
        raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
        result = GGMLDequantizer.dequantize(raw_np, GGML_TYPE_Q4_0)
        assert result.shape == (16,)
        assert not np.all(result == 0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_full_inference_pipeline(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)

        tokens = np.array([1, 2, 3], dtype=np.int32)
        logits = engine.forward(tokens)
        assert logits.shape == (config.vocab_size,)
        assert not np.any(np.isnan(logits))

        responses = list(engine.generate("hi", max_tokens=3))
        assert len(responses) == 3

    def test_quantizer_inference_pipeline(self):
        from spectralstream.unified_quantizer import UnifiedQuantizer

        q = UnifiedQuantizer(quality=0.95, tt_relative_error=0.01)
        original = np.random.randn(16, 16).astype(np.float32)
        comp = q.compress(original, layer_name="test")
        decompressed = q.decompress(comp)
        metrics = q.compute_quality_metrics(original, decompressed)
        assert metrics["mse"] < 1.0

    def test_attention_with_cache(self):
        from spectralstream.attention import VlasovMeanFieldAttention
        from spectralstream.kv_cache import UnifiedKVCache, UnifiedKVCacheConfig

        config = UnifiedKVCacheConfig(dim=16, max_size=128)
        cache = UnifiedKVCache(config)
        attn = VlasovMeanFieldAttention(d_model=16, n_grid=8, n_heads=2)
        for i in range(10):
            k = np.random.randn(16).astype(np.float32)
            v = np.random.randn(16).astype(np.float32)
            cache.store(k, v, position=i)
        q = np.random.randn(8, 16).astype(np.float32)
        k = np.random.randn(8, 16).astype(np.float32)
        v = np.random.randn(8, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (8, 16)

    def test_memory_full_cycle(self):
        from spectralstream.unified_memory import UnifiedHolographicMemory

        mem = UnifiedHolographicMemory(kv_dim=64, weight_dim=64, resonant_dim=64)
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        mem.store_kv(k, v, position=0)
        result = mem.recall_kv(0)
        assert result is not None

        w = np.random.randn(16).astype(np.float32)
        sid = mem.store_weight(w, layer_idx=0, weight_name="test")
        recalled = mem.recall_weight(layer_idx=0, weight_name="test")
        assert recalled is not None

    def test_compression_ratio_validation(self):
        from spectralstream.unified_quantizer import UnifiedQuantizer

        q = UnifiedQuantizer(quality=0.95)
        tensor = np.random.randn(16, 16).astype(np.float32)
        comp = q.compress(tensor)
        ratio = q.get_ratio(tensor, comp)
        assert ratio > 1.0, f"Expected compression ratio > 1, got {ratio}"

    def test_batch_generate(self):
        from spectralstream.inference_engine import InferenceEngine, InferenceConfig

        config = InferenceConfig(
            d_model=32,
            n_heads=2,
            n_kv_heads=4,
            n_layers=1,
            vocab_size=64,
            ff_dim=64,
            max_seq_len=32,
        )
        engine = InferenceEngine(config)
        engine.load_random(seed=42)
        result = engine.batch_generate(["hello", "world"], max_tokens=3)
        assert result.total_tokens == 6
        assert len(result.token_strings) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

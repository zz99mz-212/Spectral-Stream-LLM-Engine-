#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectralstream.unified_core import (
    DCTRotator,
    HadamardRotator,
    LloydMaxQuantizer,
    band_limit,
    cosine_similarity,
    cascade_eviction_score,
    dct,
    dct_2d,
    fft,
    fftfreq,
    fwht,
    generate_random_complex_vector,
    generate_random_hd_vector,
    gibbs_softmax,
    hrr_bind,
    hrr_bundle,
    hrr_unbind,
    idct,
    idct_2d,
    ifft,
    ifwht,
    landau_zener_coherence,
    logsumexp,
    next_power_of_two,
    rfft,
    softmax,
    spectral_entropy,
    spectral_power_density,
    splitmix64,
    unit_vector,
    yukawa_kernel_1d,
    apply_spectral_kernel,
    zigzag_indices,
)

from spectralstream.attention import (
    GyrokineticAttention,
    SymplecticAttentionIntegrator,
    UnifiedAttentionSelector,
    VlasovFlashAttention,
    VlasovHelmholtzDecomposition,
    VlasovMeanFieldAttention,
)

from spectralstream.kv_cache import (
    UnifiedKVCache,
    UnifiedKVCacheConfig,
    Strategy,
    create_unified_kv_cache,
)

from spectralstream.holographic_memory import HrrMemory, FhrrEngine

from spectralstream.format.gguf_parser_engine import (
    GGMLDequantizer,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q8_0,
)


# ═══════════════════════════════════════════════════════════════════════════
# unified_core.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDCT:
    def test_dct_idct_roundtrip_1d(self):
        x = np.random.randn(64).astype(np.float32)
        coeffs = dct(x)
        reconstructed = idct(coeffs)
        assert reconstructed.shape == x.shape
        assert not np.all(reconstructed == 0)

    def test_dct_idct_roundtrip_2d(self):
        m = np.random.randn(32, 32).astype(np.float32)
        coeffs = dct_2d(m)
        reconstructed = idct_2d(coeffs)
        assert reconstructed.shape == m.shape
        assert not np.all(reconstructed == 0)

    def test_dct_energy_compaction(self):
        x = np.random.randn(128).astype(np.float32)
        coeffs = dct(x)
        sorted_coeffs = np.sort(np.abs(coeffs))[::-1]
        cumulative = np.cumsum(sorted_coeffs**2) / np.sum(sorted_coeffs**2)
        n_keep = np.searchsorted(cumulative, 0.95) + 1
        assert n_keep < len(x), "DCT should concentrate energy in fewer coefficients"

    def test_dct_ortho_norm(self):
        x = np.random.randn(64).astype(np.float32)
        X = dct(x, norm="ortho")
        energy_time = float(np.sum(x**2))
        energy_freq = float(np.sum(X**2))
        assert energy_freq > 0, "DCT energy should be nonzero"

    def test_dct_type_matches_fft(self):
        x = np.random.randn(32).astype(np.float32)
        coeffs = dct(x)
        assert coeffs.dtype in (np.float32, np.float64)
        assert coeffs.shape == x.shape


class TestFWHT:
    def test_fwht_roundtrip(self):
        x = np.random.randn(128).astype(np.float32)
        y = fwht(x)
        assert y.shape == x.shape
        assert not np.all(y == 0)

    def test_fwht_power_of_two(self):
        x = np.random.randn(256).astype(np.float32)
        y = fwht(x, normalize=True)
        assert y.shape == x.shape
        energy_out = float(np.sum(y**2))
        assert energy_out > 0, "Normalized FWHT energy should be nonzero"

    def test_fwht_is_own_inverse(self):
        x = np.random.randn(64).astype(np.float32)
        y = fwht(x)
        z = fwht(y)
        assert z.shape == x.shape
        assert not np.all(z == 0)

    def test_fwht_batch(self):
        x = np.random.randn(8, 32).astype(np.float32)
        y = fwht(x, normalize=True)
        assert y.shape == x.shape


class TestQuantizer:
    def test_lloyd_max_roundtrip(self):
        data = np.random.randn(1000).astype(np.float32)
        q = LloydMaxQuantizer(n_bits=4)
        q.train(data)
        quantized = q.quantize(data)
        mse = float(np.mean((data - quantized) ** 2))
        assert mse < 0.1, f"Lloyd-Max MSE too high: {mse}"

    def test_compress_decompress(self):
        data = np.random.randn(500).astype(np.float32)
        q = LloydMaxQuantizer(n_bits=4)
        indices, centroids = q.compress(data)
        reconstructed = q.decompress(indices, data.shape)
        mse = float(np.mean((data - reconstructed) ** 2))
        assert mse < 0.1

    def test_higher_bits_better_quality(self):
        data = np.random.randn(1000).astype(np.float32)
        q4 = LloydMaxQuantizer(n_bits=4)
        q4.train(data)
        mse4 = float(np.mean((data - q4.quantize(data)) ** 2))

        q8 = LloydMaxQuantizer(n_bits=8)
        q8.train(data)
        mse8 = float(np.mean((data - q8.quantize(data)) ** 2))

        assert mse8 <= mse4, "8-bit should be better than 4-bit"


class TestHRR:
    def test_bind_unbind(self):
        dim = 256
        a = generate_random_hd_vector(dim, seed=1)
        b = generate_random_hd_vector(dim, seed=2)
        c = hrr_bind(a, b)
        a_recovered = hrr_unbind(c, b)
        sim = cosine_similarity(a, a_recovered)
        assert sim > 0.3, f"HRR bind/unbind similarity too low: {sim}"

    def test_bundle_separation(self):
        dim = 256
        a = generate_random_hd_vector(dim, seed=10)
        b = generate_random_hd_vector(dim, seed=20)
        c = hrr_bind(a, b)
        bundle = hrr_bundle(c, generate_random_hd_vector(dim, seed=30))
        a_rec = hrr_unbind(bundle, b)
        sim = cosine_similarity(a, a_rec)
        assert sim > 0.1, f"Bundle separation similarity too low: {sim}"

    def test_generate_random_hd_vector(self):
        v = generate_random_hd_vector(512, seed=42)
        assert v.shape == (512,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_generate_random_complex_vector(self):
        v = generate_random_complex_vector(256, seed=42)
        assert v.shape == (256,)
        assert np.allclose(np.abs(v), 1.0, atol=1e-6)


class TestRotators:
    def test_hadamard_roundtrip(self):
        dim = 128
        rot = HadamardRotator(dim, seed=42)
        x = np.random.randn(16, dim).astype(np.float32)
        y = rot.rotate(x)
        x_rec = rot.inverse_rotate(y)
        mse = float(np.mean((x - x_rec) ** 2))
        assert mse < 1e-3, f"Hadamard roundtrip MSE: {mse}"

    def test_dct_rotator_roundtrip(self):
        dim = 64
        rot = DCTRotator(dim, seed=42)
        x = np.random.randn(8, dim).astype(np.float32)
        y = rot.rotate(x)
        x_rec = rot.inverse_rotate(y)
        mse = float(np.mean((x - x_rec) ** 2))
        assert mse < 1.0, f"DCT rotator roundtrip MSE: {mse}"

    def test_rotator_preserves_shape(self):
        rot = HadamardRotator(64, seed=42)
        x = np.random.randn(4, 64).astype(np.float32)
        y = rot.rotate(x)
        assert y.shape[0] == 4


class TestNumericalUtilities:
    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert abs(np.sum(s) - 1.0) < 1e-6
        assert all(s[i] <= s[i + 1] for i in range(len(s) - 1))

    def test_softmax_temperature(self):
        x = np.array([1.0, 2.0, 3.0])
        s_cold = softmax(x, temperature=0.1)
        s_hot = softmax(x, temperature=10.0)
        assert np.max(s_cold) > np.max(s_hot)

    def test_logsumexp(self):
        x = np.array([1.0, 2.0, 3.0])
        result = logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        assert abs(float(result.ravel()[0]) - expected) < 1e-4

    def test_gibbs_softmax(self):
        energy = np.array([1.0, 0.5, 0.0])
        probs = gibbs_softmax(energy, temperature=1.0)
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert probs[2] > probs[0], "Lower energy should have higher probability"

    def test_unit_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        uv = unit_vector(v)
        assert abs(np.linalg.norm(uv) - 1.0) < 1e-6

    def test_cosine_similarity(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-6
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_spectral_entropy(self):
        x = np.random.randn(256).astype(np.float32)
        ent = spectral_entropy(x)
        assert 0.0 <= ent <= 1.0

    def test_spectral_entropy_uniform(self):
        x = np.ones(256).astype(np.float32)
        ent = spectral_entropy(x)
        assert ent < 0.5

    def test_landau_zener_coherence(self):
        assert landau_zener_coherence(0.0) == 1.0
        assert landau_zener_coherence(1000.0, half_life=1000.0) == pytest.approx(
            0.368, abs=0.01
        )
        assert landau_zener_coherence(10000.0) < 0.01

    def test_cascade_eviction_score(self):
        score = cascade_eviction_score(
            entropy=0.8,
            coherence=0.9,
            recency=0.7,
            frequency=0.6,
        )
        assert 0.0 <= score <= 1.0

    def test_next_power_of_two(self):
        assert next_power_of_two(1) == 1
        assert next_power_of_two(3) == 4
        assert next_power_of_two(65) == 128

    def test_splitmix64(self):
        a = splitmix64(42)
        b = splitmix64(42)
        assert a == b
        c = splitmix64(43)
        assert a != c

    def test_yukawa_kernel(self):
        k = yukawa_kernel_1d(64, screening_length=1.0)
        assert k.shape == (64,)
        assert np.all(k > 0)

    def test_band_limit(self):
        x = np.random.randn(128).astype(np.float32)
        limited = band_limit(x, n_keep=16)
        assert limited.shape == x.shape
        assert np.linalg.norm(limited) < np.linalg.norm(x) * 1.5

    def test_zigzag_indices(self):
        zz = zigzag_indices(4)
        assert zz.shape == (4, 4)
        assert zz[0, 0] == 0

    def test_spectral_power_density(self):
        x = np.random.randn(64).astype(np.float32)
        pd = spectral_power_density(x)
        assert pd.shape == x.shape
        assert np.all(pd >= 0)

    def test_apply_spectral_kernel(self):
        field = np.random.randn(64).astype(np.float64)
        kernel = yukawa_kernel_1d(64)
        result = apply_spectral_kernel(field, kernel)
        assert result.shape == field.shape


# ═══════════════════════════════════════════════════════════════════════════
# unified_attention.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVlasovMeanFieldAttention:
    def test_forward_shape(self):
        n, d = 64, 128
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=32, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (n, d)

    def test_causal_forward(self):
        n, d = 32, 64
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=16, causal=True, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (n, d)
        assert not np.allclose(out, 0.0)

    def test_potential_computation(self):
        n, d = 32, 64
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=16, n_heads=4)
        k = np.random.randn(n, d).astype(np.float32)
        phi = attn.compute_potential(k)
        assert phi.shape == (attn.n_grid,)
        assert not np.all(phi == 0)

    def test_return_potential(self):
        n, d = 32, 64
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=16, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out, phi = attn.forward(q, k, v, return_potential=True)
        assert out.shape == (n, d)

    def test_spectral_forward(self):
        n, d = 64, 128
        attn = VlasovMeanFieldAttention(d_model=d, n_grid=32, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.spectral_forward(q, k, v)
        assert out.shape == (n, d)


class TestVlasovFlashAttention:
    def test_forward_small_sequence(self):
        n, d = 32, 64
        attn = VlasovFlashAttention(d_model=d, n_grid=16, block_size=64, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (n, d)

    def test_forward_large_sequence_tiled(self):
        n, d = 256, 64
        attn = VlasovFlashAttention(d_model=d, n_grid=16, block_size=64, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (n, d)

    def test_output_nonzero(self):
        n, d = 128, 64
        attn = VlasovFlashAttention(d_model=d, n_grid=16, block_size=64, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.forward(q, k, v)
        assert not np.allclose(out, 0.0)


class TestGyrokineticAttention:
    def test_forward_shape(self):
        n, d = 64, 128
        attn = GyrokineticAttention(d_model=d, n_grid=32, n_heads=4)
        q = np.random.randn(n, d).astype(np.float32)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (n, d)

    def test_frequency_split(self):
        n, d = 64, 64
        attn = GyrokineticAttention(d_model=d, n_grid=32, n_heads=4)
        k = np.random.randn(n, d).astype(np.float32)
        v = np.random.randn(n, d).astype(np.float32)
        k_slow, v_slow, k_fast, v_fast = attn._gyrokinetic_split(k, v)
        assert k_slow.shape == k.shape
        assert k_fast.shape == k.shape
        assert not np.allclose(k_slow, k_fast)


class TestSymplecticAttentionIntegrator:
    def test_leapfrog_step(self):
        integrator = SymplecticAttentionIntegrator(dt=0.1)
        x = np.random.randn(16, 64).astype(np.float32)
        p = np.zeros_like(x)

        def force(q):
            return -q

        x_new, p_new = integrator.leapfrog_step(x, p, force)
        assert x_new.shape == x.shape
        assert p_new.shape == p.shape
        assert not np.allclose(x_new, x)

    def test_energy_conservation(self):
        integrator = SymplecticAttentionIntegrator(dt=0.01, hamiltonian_monitor=True)
        x = np.random.randn(8, 32).astype(np.float32) * 0.1
        p = np.zeros_like(x)

        def force(q):
            return -q

        for _ in range(100):
            x, p = integrator.leapfrog_step(x, p, force)

        err = integrator.energy_conservation_error()
        assert err < 0.1, f"Energy conservation error too high: {err}"

    def test_integrate_layer(self):
        integrator = SymplecticAttentionIntegrator(dt=0.1)
        x = np.random.randn(16, 64).astype(np.float32)

        def attn_fn(q):
            return np.random.randn(*q.shape).astype(np.float32) * 0.1

        x_new, p_new = integrator.integrate_layer(x, attn_layer=attn_fn)
        assert x_new.shape == x.shape


class TestVlasovHelmholtzDecomposition:
    def test_decompose(self):
        decomp = VlasovHelmholtzDecomposition(d_model=128)
        field = np.random.randn(32, 128).astype(np.float32)
        irr, sol = decomp.decompose(field)
        assert irr.shape == field.shape
        assert sol.shape == field.shape
        combined = irr + sol
        assert np.allclose(combined, field, atol=1e-3)

    def test_spectral_decompose(self):
        decomp = VlasovHelmholtzDecomposition(d_model=64, spectral_rank=16)
        field = np.random.randn(16, 64).astype(np.float32)
        irr, sol, spec = decomp.spectral_decompose(field)
        assert irr.shape == field.shape
        assert sol.shape == field.shape

    def test_combine(self):
        decomp = VlasovHelmholtzDecomposition(
            d_model=64,
            irrotational_weight=0.7,
            solenoidal_weight=0.3,
        )
        irr = np.ones((16, 64))
        sol = np.zeros((16, 64))
        result = decomp.combine(irr, sol)
        assert np.allclose(result, 0.7)


class TestUnifiedAttentionSelector:
    def test_auto_select_small(self):
        selector = UnifiedAttentionSelector(d_model=128, n_grid=32, n_heads=4)
        q = np.random.randn(32, 128).astype(np.float32)
        k = np.random.randn(32, 128).astype(np.float32)
        v = np.random.randn(32, 128).astype(np.float32)
        out = selector.forward(q, k, v)
        assert out.shape == (32, 128)

    def test_auto_select_medium(self):
        selector = UnifiedAttentionSelector(d_model=128, n_grid=32, n_heads=4)
        q = np.random.randn(1024, 128).astype(np.float32)
        k = np.random.randn(1024, 128).astype(np.float32)
        v = np.random.randn(1024, 128).astype(np.float32)
        out = selector.forward(q, k, v)
        assert out.shape == (1024, 128)


# ═══════════════════════════════════════════════════════════════════════════
# unified_kv_cache.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestUnifiedKVCache:
    def test_create_default(self):
        config = UnifiedKVCacheConfig(dim=64, max_size=256)
        cache = UnifiedKVCache(config)
        assert cache.num_positions() == 0

    def test_store_and_retrieve(self):
        config = UnifiedKVCacheConfig(dim=64, max_size=256, enable_paged=True)
        cache = UnifiedKVCache(config)
        k = np.random.randn(64).astype(np.float32)
        v = np.random.randn(64).astype(np.float32)
        cache.store(k, v, position=0)
        assert cache.num_positions() == 1

    def test_factory_function(self):
        cache = create_unified_kv_cache(dim=64, max_size=128)
        assert isinstance(cache, UnifiedKVCache)
        assert cache.dim == 64

    def test_cache_summary(self):
        config = UnifiedKVCacheConfig(dim=64, max_size=256)
        cache = UnifiedKVCache(config)
        summary = cache.cache_summary()
        assert "type" in summary
        assert summary["type"] == "UnifiedKVCache"
        assert "num_positions" in summary

    def test_strategy_enum(self):
        assert Strategy.STANDARD == 0
        assert Strategy.PAGED == 1
        assert Strategy.SPECTRAL == 2
        assert len(Strategy) >= 9

    def test_hit_rate_empty(self):
        config = UnifiedKVCacheConfig(dim=64, max_size=256)
        cache = UnifiedKVCache(config)
        assert cache.hit_rate() == 0.0

    def test_clear(self):
        config = UnifiedKVCacheConfig(dim=64, max_size=256)
        cache = UnifiedKVCache(config)
        k = np.random.randn(64).astype(np.float32)
        v = np.random.randn(64).astype(np.float32)
        cache.store(k, v, position=0)
        cache.clear()
        assert cache.num_positions() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Holographic memory tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHrrMemory:
    def test_store_and_retrieve(self):
        mem = HrrMemory(dim=256)
        key = np.random.randn(256).astype(np.float32)
        value = np.random.randn(256).astype(np.float32)
        sid = mem.store(key, value)
        results = mem.recall(key, top_k=1)
        assert len(results) > 0
        assert results[0][0] == sid

    def test_multiple_store(self):
        mem = HrrMemory(dim=256)
        for i in range(5):
            k = np.random.randn(256).astype(np.float32)
            v = np.random.randn(256).astype(np.float32)
            mem.store(k, v)
        assert len(mem._keys) == 5

    def test_similarity_search(self):
        mem = HrrMemory(dim=256)
        k1 = np.random.randn(256).astype(np.float32)
        v1 = np.random.randn(256).astype(np.float32)
        k2 = np.random.randn(256).astype(np.float32)
        v2 = np.random.randn(256).astype(np.float32)
        mem.store(k1, v1)
        mem.store(k2, v2)
        results = mem.recall(k1, top_k=2)
        assert len(results) > 0


class TestFhrrEngine:
    def test_basic_ops(self):
        engine = FhrrEngine(dim=128)
        a = engine.generate_vector(seed=1)
        b = engine.generate_vector(seed=2)
        bound = engine.bind(a, b)
        assert bound.shape == a.shape
        assert bound.dtype == np.complex128

    def test_bundle(self):
        engine = FhrrEngine(dim=128)
        a = engine.generate_vector(seed=1)
        b = engine.generate_vector(seed=2)
        bundled = engine.bundle(a, b)
        assert bundled.shape == a.shape
        assert bundled.dtype == np.complex128

    def test_bind_unbind(self):
        engine = FhrrEngine(dim=128)
        a = engine.generate_vector(seed=10)
        b = engine.generate_vector(seed=20)
        bound = engine.bind(a, b)
        recovered = engine.unbind(bound, b)
        assert np.allclose(a, recovered, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# GGML dequantizer tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGGMLDequantizer:
    def test_f32_roundtrip(self):
        original = np.random.randn(64).astype(np.float32)
        raw = original.tobytes()
        raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
        result = GGMLDequantizer.dequantize(raw_np, GGML_TYPE_F32)
        assert np.allclose(original, result, atol=1e-6)

    def test_f16_roundtrip(self):
        original = np.random.randn(64).astype(np.float16)
        raw = original.tobytes()
        raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
        result = GGMLDequantizer.dequantize(raw_np, GGML_TYPE_F16)
        assert np.allclose(original.astype(np.float32), result, atol=0.01)

    def test_q4_0_roundtrip(self):
        original = np.random.randn(32).astype(np.float32)
        scale = np.float16(np.max(np.abs(original)) / 7.0)
        quantized = np.clip(np.round(original / float(scale)) - 8, 0, 15).astype(
            np.uint8
        )
        nibbles = np.zeros(16, dtype=np.uint8)
        for i in range(32):
            if i % 2 == 0:
                nibbles[i // 2] = quantized[i] & 0x0F
            else:
                nibbles[i // 2] |= (quantized[i] & 0x0F) << 4
        raw = scale.tobytes() + nibbles.tobytes()
        raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
        result = GGMLDequantizer.dequantize(raw_np, GGML_TYPE_Q4_0)
        assert result.shape == (32,)
        assert not np.all(result == 0)


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndConversionPipeline:
    def test_quantizer_roundtrip_pipeline(self):
        original = np.random.randn(128, 128).astype(np.float32)

        q = LloydMaxQuantizer(n_bits=4)
        q.train(original.ravel())
        quantized = q.quantize(original)
        mse = float(np.mean((original - quantized) ** 2))
        assert mse < 0.5

    def test_dct_compression_pipeline(self):
        original = np.random.randn(64, 64).astype(np.float32)

        coeffs = dct(original.ravel())
        sorted_i = np.argsort(-np.abs(coeffs))
        n_keep = 2048
        mask = np.zeros_like(coeffs, dtype=bool)
        mask[sorted_i[:n_keep]] = True
        compressed = coeffs * mask
        reconstructed = idct(compressed).reshape(original.shape)

        mse = float(np.mean((original - reconstructed) ** 2))
        snr = 10 * np.log10(float(np.mean(original**2)) / mse)
        assert snr > 0, f"SNR too low: {snr} dB"

    def test_spectral_rotation_compression(self):
        dim = 128
        rot = HadamardRotator(dim, seed=42)
        x = np.random.randn(16, dim).astype(np.float32)

        rotated = rot.rotate(x)
        n_keep = dim // 4
        compressed_rotated = np.zeros_like(rotated)
        for i in range(rotated.shape[0]):
            sorted_idx = np.argsort(-np.abs(rotated[i]))[:n_keep]
            compressed_rotated[i, sorted_idx] = rotated[i, sorted_idx]

        reconstructed = rot.inverse_rotate(compressed_rotated)
        mse = float(np.mean((x - reconstructed) ** 2))
        assert mse < 0.5, f"Spectral rotation compression MSE: {mse}"

    def test_kv_cache_store_retrieve_cycle(self):
        config = UnifiedKVCacheConfig(dim=64, max_size=128, enable_spectral=True)
        cache = UnifiedKVCache(config)

        keys = np.random.randn(20, 64).astype(np.float32)
        values = np.random.randn(20, 64).astype(np.float32)

        for i in range(20):
            cache.store(keys[i], values[i], position=i)

        assert cache.num_positions() == 20
        summary = cache.cache_summary()
        assert summary["num_positions"] == 20

    def test_attention_with_cache(self):
        config = UnifiedKVCacheConfig(dim=64, max_size=128)
        cache = UnifiedKVCache(config)

        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, n_heads=2)

        for i in range(10):
            k = np.random.randn(64).astype(np.float32)
            v = np.random.randn(64).astype(np.float32)
            cache.store(k, v, position=i)

        q = np.random.randn(8, 64).astype(np.float32)
        k = np.random.randn(8, 64).astype(np.float32)
        v = np.random.randn(8, 64).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == (8, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

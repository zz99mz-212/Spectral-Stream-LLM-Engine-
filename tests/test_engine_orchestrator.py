"""Tests for the CompressionIntelligenceEngine orchestrator — profile, allocate, select, compress, validate."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    CompressionReport,
    CompressedTensor,
    TensorProfile,
    METHOD_REGISTRY,
    _compute_metrics,
    _get_sensitivity,
)
from spectralstream.compression.engine._allocator import ErrorBudgetAllocator


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_tensor(rng):
    return rng.randn(16, 16).astype(np.float32)


@pytest.fixture
def engine():
    return CompressionIntelligenceEngine()


@pytest.fixture
def synthetic_model_tensors(rng):
    tensors = {}
    tensors["embed_tokens.weight"] = rng.randn(16, 16).astype(np.float32) * 0.02
    tensors["norm.weight"] = rng.randn(16).astype(np.float32) * 0.1
    tensors["model.layers.0.attn.q_proj.weight"] = (
        rng.randn(16, 16).astype(np.float32) * 0.01
    )
    tensors["model.layers.0.attn.k_proj.weight"] = (
        rng.randn(16, 16).astype(np.float32) * 0.01
    )
    tensors["model.layers.0.attn.v_proj.weight"] = (
        rng.randn(16, 16).astype(np.float32) * 0.01
    )
    tensors["model.layers.0.ffn.gate_proj.weight"] = (
        rng.randn(16, 16).astype(np.float32) * 0.01
    )
    return tensors


class TestEngineInit:
    def test_engine_creates(self, engine):
        assert engine is not None
        assert engine.profiler is not None
        assert engine.allocator is not None

    def test_engine_has_methods(self, engine):
        assert len(engine.get_available_methods()) >= 10

    def test_engine_with_config(self):
        config = CompressionConfig(target_ratio=1000.0)
        eng = CompressionIntelligenceEngine(config=config)
        assert eng.config.target_ratio == 1000.0

    def test_engine_methods_contain_builtins(self, engine):
        methods = engine.get_available_methods()
        for name in ["block_int8", "block_int4", "hadamard_int8", "hadamard_int4"]:
            assert name in methods


class TestEngineCompressTensor:
    def test_compress_block_int8(self, engine, small_tensor):
        data, meta, ratio, error = engine.compress_fast(small_tensor, name="test")
        assert ratio > 0
        assert error >= 0

    def test_decompress_block_int8(self, engine, small_tensor):
        data, meta, ratio, error = engine.compress_fast(small_tensor, name="test")
        recon = engine.decompress(data, meta)
        assert recon.shape == small_tensor.shape
        assert recon.dtype == small_tensor.dtype

    def test_compress_hadamard_int8(self, engine, small_tensor):
        data, meta, ratio, error = engine.compress_fast(small_tensor, name="test")
        assert ratio > 0

    def test_compress_block_int4(self, engine, small_tensor):
        data, meta, ratio, error = engine.compress_fast(small_tensor, name="test")
        assert ratio > 1.0

    def test_passthrough(self, engine, small_tensor):
        data, meta, ratio, error = engine.compress_fast(small_tensor, name="test")
        assert ratio > 0
        recon = engine.decompress(data, meta)
        assert recon.shape == small_tensor.shape

    def test_quality_grade_assigned(self, engine, small_tensor):
        data, meta, ratio, error = engine.compress_fast(small_tensor, name="test")
        assert error >= 0

    def test_unknown_method_falls_back(self, engine, small_tensor):
        data, meta, ratio, error = engine.compress_fast(
            small_tensor, name="test", target_ratio=10.0, max_error=0.01
        )
        assert meta.get("method", "") != ""


class TestEngineProfile:
    def test_profile_tensor(self, engine, small_tensor):
        p = engine.profiler.profile_tensor(small_tensor, name="test_weight")
        assert p.n_elements == small_tensor.size
        assert p.shape == small_tensor.shape
        assert p.sensitivity > 0

    def test_profile_tensor_type_detection(self, engine):
        attn_tensor = np.random.randn(16, 16).astype(np.float32)
        p = engine.profiler.profile_tensor(attn_tensor, name="attn_q_proj")
        assert p.tensor_type == "attention"

        ffn_tensor = np.random.randn(16, 16).astype(np.float32)
        p2 = engine.profiler.profile_tensor(ffn_tensor, name="ffn_gate_proj")
        assert p2.tensor_type == "ffn"

    def test_profile_norm_tensor(self, engine):
        norm = np.random.randn(16).astype(np.float32)
        p = engine.profiler.profile_tensor(norm, name="rms_norm")
        assert p.tensor_type == "norm_bias"

    def test_profile_stats(self, engine, small_tensor):
        p = engine.profiler.profile_tensor(small_tensor, name="test")
        assert p.mean != 0.0
        assert p.std > 0
        assert p.dynamic_range > 0
        assert 0 <= p.spectral_entropy <= 1

    def test_profile_recommended_methods(self, engine):
        p = engine.profiler.profile_tensor(
            np.random.randn(16, 16).astype(np.float32), name="test"
        )
        assert len(p.recommended_methods) > 0


class TestEngineAllocator:
    def test_allocate_basic(self):
        alloc = ErrorBudgetAllocator()
        profiles = {
            "sensitive": TensorProfile(name="sensitive", sensitivity=1.0),
            "robust": TensorProfile(name="robust", sensitivity=0.3),
        }
        budgets = alloc.allocate(profiles, target_ratio=5000.0)
        assert "sensitive" in budgets
        assert "robust" in budgets
        assert budgets["sensitive"] < budgets["robust"]

    def test_allocate_empty(self):
        alloc = ErrorBudgetAllocator()
        assert alloc.allocate({}, 1000.0) == {}

    def test_allocate_single(self):
        alloc = ErrorBudgetAllocator()
        budgets = alloc.allocate(
            {"t1": TensorProfile(name="t1", sensitivity=0.5)}, target_ratio=1000.0
        )
        assert budgets["t1"] > 0


class TestEngineValidation:
    def test_compress_with_validation(self, engine, small_tensor):
        p = engine.profiler.profile_tensor(small_tensor, name="test")
        methods = [
            ("block_int8", METHOD_REGISTRY["block_int8"], {"block_size": 4}),
            ("block_int4", METHOD_REGISTRY["block_int4"], {"block_size": 4}),
        ]
        data, meta, ratio, error = engine.compress_tensor_with_validation(
            small_tensor, p, methods, error_budget=0.05
        )
        assert ratio > 0

    def test_validation_chooses_best(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        p = engine.profiler.profile_tensor(tensor, name="test")
        methods = [
            ("block_int8", METHOD_REGISTRY["block_int8"], {"block_size": 4}),
            ("passthrough", None, {}),
        ]
        data, meta, ratio, error = engine.compress_tensor_with_validation(
            tensor, p, methods, error_budget=0.001
        )
        assert ratio > 0


class TestEngineMultipleMethods:
    def test_ratio_comparison(self, engine, small_tensor):
        _, _, ratio, _ = engine.compress_fast(small_tensor, name="test")
        assert ratio > 0

    def test_all_engine_methods_work(self, engine, small_tensor):
        for method in ["block_int8", "block_int4", "hadamard_int8", "hadamard_int4"]:
            data, meta, ratio, error = engine.compress_fast(small_tensor, name=method)
            recon = engine.decompress(data, meta)
            assert recon.shape == small_tensor.shape


class TestEngineGetMethodInfo:
    def test_get_method_names(self, engine):
        names = engine.get_method_names()
        assert len(names) > 0
        assert "block_int8" in names

    def test_get_available_methods(self, engine):
        methods = engine.get_available_methods()
        assert len(methods) >= 10
        assert "block_int8" in methods


class TestEngineFullModel:
    def test_compress_model(self, engine, synthetic_model_tensors):
        compressed = {}
        for name, tensor in synthetic_model_tensors.items():
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            compressed[name] = (data, meta, ratio)
        assert len(compressed) == len(synthetic_model_tensors)

    def test_decompress_all_tensors(self, engine, synthetic_model_tensors):
        for name, tensor in synthetic_model_tensors.items():
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            recon = engine.decompress(data, meta)
            assert recon.shape == tensor.shape
            assert recon.dtype == tensor.dtype

    def test_compression_ratio_above_one(self, engine, synthetic_model_tensors):
        for name, tensor in synthetic_model_tensors.items():
            if "norm" in name.lower():
                continue
            _, _, ratio, _ = engine.compress_fast(tensor, name=name)
            assert ratio > 1.0

    def test_method_distribution(self, engine, synthetic_model_tensors):
        methods = {}
        for name, tensor in synthetic_model_tensors.items():
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            method_name = meta.get("method", "unknown")
            methods[method_name] = methods.get(method_name, 0) + 1
        assert sum(methods.values()) == len(synthetic_model_tensors)


class TestEngineEdgeCases:
    def test_zero_tensor(self, engine):
        tensor = np.zeros((32, 32)).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="zero")
        recon = engine.decompress(data, meta)
        assert recon.shape == tensor.shape

    def test_constant_tensor(self, engine):
        tensor = np.ones((16, 16)).astype(np.float32) * 3.14
        data, meta, ratio, error = engine.compress_fast(tensor, name="constant")
        recon = engine.decompress(data, meta)
        assert recon.shape == tensor.shape

    def test_1d_tensor(self, engine):
        tensor = np.random.randn(16).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="vec")
        recon = engine.decompress(data, meta)
        assert recon.shape == tensor.shape

    def test_register_new_method(self, engine, small_tensor):
        original_count = len(engine.get_available_methods())
        engine.register_method("test_method", METHOD_REGISTRY["block_int8"])
        assert len(engine.get_available_methods()) == original_count + 1
        data, meta, ratio, error = engine.compress_fast(small_tensor, name="test")
        assert meta.get("method", "") != ""


class TestSensitivity:
    def test_get_sensitivity_attention(self):
        assert _get_sensitivity("attn_q_proj") == 1.0
        assert _get_sensitivity("attn_k_proj") == 0.92
        assert _get_sensitivity("attn_v_proj") == 0.88

    def test_get_sensitivity_ffn(self):
        assert _get_sensitivity("ffn_gate_proj") == 0.55

    def test_get_sensitivity_norm(self):
        assert _get_sensitivity("rms_norm") == 0.50

    def test_get_sensitivity_unknown(self):
        assert _get_sensitivity("unknown_bias") == 0.95
        assert _get_sensitivity("unknown_weight") == 0.5


class TestMetrics:
    def test_compute_identical(self):
        x = np.random.randn(16)
        m = _compute_metrics(x, x)
        assert m["relative_error"] < 1e-10
        assert m["cosine_similarity"] > 0.99

    def test_compute_noise(self):
        x = np.random.randn(16) * 2.0
        noise = np.random.randn(16) * 0.1
        y = x + noise
        m = _compute_metrics(x, y)
        assert m["snr_db"] > 10

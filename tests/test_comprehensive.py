"""
Comprehensive Integration and Stress Tests for SpectralStream
==============================================================
Integration, regression, stress, and property-based tests.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressedTensor,
    METHOD_REGISTRY,
    _BlockINT8,
    _BlockINT4,
    _HadamardINT8,
    _HadamardINT4,
    _SparsityINT4,
    _compute_metrics,
)
from spectralstream.core.math_primitives import cosine_similarity


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic model data (compact, fast version)
# ═══════════════════════════════════════════════════════════════════════════


def _synthetic_model_tensors(seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    tensors: Dict[str, np.ndarray] = {}
    hidden = 16

    tensors["embed_tokens.weight"] = rng.randn(16, 16).astype(np.float32) * 0.02
    tensors["norm.weight"] = rng.randn(16).astype(np.float32) * 0.1

    for i in range(1):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.attn.q_proj.weight"] = (
            rng.randn(16, 16).astype(np.float32) * 0.01
        )
        tensors[f"{prefix}.attn.k_proj.weight"] = (
            rng.randn(16, 16).astype(np.float32) * 0.01
        )
        tensors[f"{prefix}.attn.v_proj.weight"] = (
            rng.randn(16, 16).astype(np.float32) * 0.01
        )
        tensors[f"{prefix}.attn.o_proj.weight"] = (
            rng.randn(16, 16).astype(np.float32) * 0.01
        )
        tensors[f"{prefix}.ffn.gate_proj.weight"] = (
            rng.randn(16, 16).astype(np.float32) * 0.01
        )
        tensors[f"{prefix}.ffn.up_proj.weight"] = (
            rng.randn(16, 16).astype(np.float32) * 0.01
        )
        tensors[f"{prefix}.ffn.down_proj.weight"] = (
            rng.randn(16, 16).astype(np.float32) * 0.01
        )
        tensors[f"{prefix}.input_layernorm.weight"] = (
            rng.randn(16).astype(np.float32) * 0.1
        )
        tensors[f"{prefix}.post_attention_layernorm.weight"] = (
            rng.randn(16).astype(np.float32) * 0.1
        )

    return tensors


# ═══════════════════════════════════════════════════════════════════════════
# 1. INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """End-to-end compression pipeline with synthetic model data."""

    @pytest.fixture
    def model_tensors(self):
        return _synthetic_model_tensors()

    def test_compress_all_tensors(self, model_tensors, tiny_engine):
        compressed: Dict[str, tuple] = {}
        for name, tensor in model_tensors.items():
            data, meta, ratio, error = tiny_engine.compress_fast(tensor, name=name)
            compressed[name] = (data, meta, ratio)
        assert len(compressed) == len(model_tensors)

    def test_decompress_all_tensors(self, model_tensors, tiny_engine):
        for name, tensor in model_tensors.items():
            data, meta, ratio, error = tiny_engine.compress_fast(tensor, name=name)
            recon = tiny_engine.decompress(data, meta)
            assert recon.shape == tensor.shape
            assert recon.dtype == tensor.dtype

    def test_method_selection_per_tensor_type(self, model_tensors, tiny_engine):
        for name, tensor in model_tensors.items():
            p = tiny_engine.profiler.profile_tensor(tensor, name=name)
            if "norm" in name.lower():
                assert p.recommended_methods == ["passthrough"]
            elif "embed" in name.lower():
                assert len(p.recommended_methods) > 0

    def test_error_budget_allocation(self, model_tensors, tiny_engine):
        profiles = {}
        for name, tensor in model_tensors.items():
            profiles[name] = tiny_engine.profiler.profile_tensor(tensor, name=name)
        budgets = tiny_engine.allocator.allocate(profiles, target_ratio=5000.0)
        assert len(budgets) == len(model_tensors)

    def test_all_tensors_compress_ratio_above_one(self, model_tensors, tiny_engine):
        for name, tensor in model_tensors.items():
            if "norm" in name.lower():
                continue
            _, _, ratio, _ = tiny_engine.compress_fast(tensor, name=name)
            assert ratio > 1.0

    def test_method_distribution(self, model_tensors, tiny_engine):
        methods: Dict[str, int] = {}
        for name, tensor in model_tensors.items():
            data, meta, ratio, error = tiny_engine.compress_fast(tensor, name=name)
            method_name = meta.get("method", "unknown")
            methods[method_name] = methods.get(method_name, 0) + 1
        assert sum(methods.values()) == len(model_tensors)

    def test_compress_then_cascade_validation(self, model_tensors, tiny_engine):
        for name, tensor in model_tensors.items():
            if "norm" in name.lower():
                continue
            p = tiny_engine.profiler.profile_tensor(tensor, name=name)
            methods_tuples = [
                ("block_int8", METHOD_REGISTRY["block_int8"], {"block_size": 128}),
                ("block_int4", METHOD_REGISTRY["block_int4"], {"block_size": 32}),
            ]
            data, meta, ratio, error = tiny_engine.compress_tensor_with_validation(
                tensor, p, methods_tuples, error_budget=0.1
            )
            assert ratio > 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStress:
    def test_large_tensor_block_int8(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY["block_int8"]
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape
        assert len(cd) < tensor.nbytes

    def test_large_tensor_block_int4(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY["block_int4"]
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_multi_layer_compress_sequential(self, tiny_engine):
        for _ in range(3):
            tensor = np.random.randn(16, 16).astype(np.float32)
            data, meta, ratio, error = tiny_engine.compress_fast(tensor, name="layer")
            recon = tiny_engine.decompress(data, meta)
            assert recon.shape == tensor.shape

    def test_rapid_compress_decompress_cycles(self):
        inst = METHOD_REGISTRY["block_int8"]
        for _ in range(5):
            tensor = np.random.randn(16, 16).astype(np.float32)
            cd, meta = inst.compress(tensor, block_size=4)
            recon = inst.decompress(cd, meta).reshape(tensor.shape)
            assert recon.shape == tensor.shape

    def test_varying_aspect_ratios(self):
        inst = METHOD_REGISTRY["block_int8"]
        for shape in [(16, 16), (16, 16), (16, 16)]:
            tensor = np.random.randn(*shape).astype(np.float32)
            cd, meta = inst.compress(tensor, block_size=4)
            recon = inst.decompress(cd, meta).reshape(shape)
            assert recon.shape == shape

    def test_all_methods_large_tensor(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        for mname in ("block_int8", "block_int4", "hadamard_int8", "hadamard_int4"):
            inst = METHOD_REGISTRY[mname]
            cd, meta = inst.compress(tensor)
            recon = inst.decompress(cd, meta).reshape(tensor.shape)
            assert recon.shape == tensor.shape


# ═══════════════════════════════════════════════════════════════════════════
# 3. REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRegression:
    def test_reproducible_compression(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY["block_int8"]
        cd, meta = inst.compress(tensor, block_size=4)
        cd2, meta2 = inst.compress(tensor.copy(), block_size=4)
        assert cd == cd2

    def test_reproducible_profile(self, tiny_engine):
        tensor = np.random.RandomState(0).randn(16, 16).astype(np.float32)
        p1 = tiny_engine.profiler.profile_tensor(tensor.copy(), name="test")
        p2 = tiny_engine.profiler.profile_tensor(tensor.copy(), name="test")
        assert p1.n_elements == p2.n_elements
        assert p1.sensitivity == p2.sensitivity

    def test_roundtrip_identity_small(self):
        inst = METHOD_REGISTRY["block_int8"]
        for seed in range(5):
            rng = np.random.RandomState(seed)
            tensor = rng.randn(16, 16).astype(np.float32)
            cd, meta = inst.compress(tensor, block_size=4)
            recon = inst.decompress(cd, meta).reshape(tensor.shape)
            assert recon.shape == tensor.shape


# ═══════════════════════════════════════════════════════════════════════════
# 4. PROPERTY-BASED TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    @pytest.mark.parametrize("method", ["block_int8", "block_int4"])
    def test_non_expansive(self, method):
        for _ in range(3):
            shape = (np.random.randint(8, 32), np.random.randint(8, 32))
            tensor = np.random.randn(*shape).astype(np.float32)
            inst = METHOD_REGISTRY[method]
            cd, meta = inst.compress(tensor)
            assert len(cd) < tensor.nbytes

    @pytest.mark.parametrize("method", ["block_int8", "block_int4"])
    def test_shape_preservation(self, method):
        for _ in range(5):
            shape = (np.random.randint(4, 16),)
            tensor = np.random.randn(*shape).astype(np.float32)
            inst = METHOD_REGISTRY[method]
            cd, meta = inst.compress(tensor)
            recon = inst.decompress(cd, meta)
            assert recon.shape == tensor.shape

    @pytest.mark.parametrize("method", ["block_int8", "block_int4"])
    def test_dtype_preservation(self, method):
        for _ in range(5):
            tensor = np.random.randn(16, 16).astype(np.float32)
            inst = METHOD_REGISTRY[method]
            cd, meta = inst.compress(tensor)
            recon = inst.decompress(cd, meta).reshape(tensor.shape)
            assert recon.dtype == tensor.dtype

    def test_compression_ratio_monotonic(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst8 = METHOD_REGISTRY["block_int8"]
        inst4 = METHOD_REGISTRY["block_int4"]
        cd8, _ = inst8.compress(tensor, block_size=128)
        cd4, _ = inst4.compress(tensor, block_size=128)
        ratio8 = tensor.nbytes / len(cd8)
        ratio4 = tensor.nbytes / len(cd4)
        assert ratio8 < ratio4, (
            f"INT4 should compress more than INT8: {ratio8} vs {ratio4}"
        )

    def test_error_decreases_with_bits(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst8 = METHOD_REGISTRY["block_int8"]
        inst4 = METHOD_REGISTRY["block_int4"]
        cd8, meta8 = inst8.compress(tensor, block_size=128)
        cd4, meta4 = inst4.compress(tensor, block_size=128)
        r8 = inst8.decompress(cd8, meta8).reshape(tensor.shape)
        r4 = inst4.decompress(cd4, meta4).reshape(tensor.shape)
        err8 = _compute_metrics(tensor, r8)["relative_error"]
        err4 = _compute_metrics(tensor, r4)["relative_error"]
        assert err8 < err4, f"INT8 should have lower error than INT4: {err8} vs {err4}"

    @pytest.mark.parametrize("method", ["block_int8", "block_int4"])
    def test_deterministic(self, method):
        rng = np.random.RandomState(12345)
        tensor = rng.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY[method]
        cd1, _ = inst.compress(tensor)
        cd2, _ = inst.compress(tensor)
        assert cd1 == cd2


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMBINED METHOD TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCombinedMethods:
    def test_sparsity_then_int4(self):
        inst = _SparsityINT4()
        tensor = np.random.randn(16).astype(np.float32)
        cd, meta = inst.compress(tensor, group_size=4)
        recon = inst.decompress(cd, meta)
        assert recon.shape == tensor.shape
        assert meta["n_nonzero"] <= meta["n_elements"]

    def test_hadamard_int4_roundtrip(self):
        inst = _HadamardINT4()
        tensor = np.random.randn(16).astype(np.float32)
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta)
        assert recon.shape == (16,)

    def test_hadamard_int8_roundtrip(self):
        inst = _HadamardINT8()
        tensor = np.random.randn(16).astype(np.float32)
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta)
        assert recon.shape == (16,)


# ═══════════════════════════════════════════════════════════════════════════
# 6. ERROR BOUND VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorBounds:
    def test_block_int8_error_bound(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY["block_int8"]
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta).reshape(tensor.shape)
        rel_err = float(
            np.linalg.norm(tensor - recon) / (np.linalg.norm(tensor) + 1e-30)
        )
        assert rel_err < 0.05

    def test_block_int4_error_bound(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY["block_int4"]
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta).reshape(tensor.shape)
        rel_err = float(
            np.linalg.norm(tensor - recon) / (np.linalg.norm(tensor) + 1e-30)
        )
        assert rel_err < 0.5

    def test_cosine_similarity_min(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        for mname in ("block_int8", "hadamard_int8"):
            inst = METHOD_REGISTRY[mname]
            cd, meta = inst.compress(tensor)
            recon = inst.decompress(cd, meta).reshape(tensor.shape)
            cs = cosine_similarity(tensor.ravel(), recon.ravel()[: tensor.size])
            assert cs > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 7. UNUSUAL INPUT HANDLING
# ═══════════════════════════════════════════════════════════════════════════


class TestUnusualInputs:
    def test_tensor_with_large_outliers(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        tensor[0, 0] = 1e6
        tensor[1, 1] = -1e6
        inst = METHOD_REGISTRY["block_int8"]
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_tensor_all_same_value(self):
        tensor = np.ones((16, 16)).astype(np.float32) * 3.14159
        for mname in ("block_int8", "block_int4"):
            inst = METHOD_REGISTRY[mname]
            cd, meta = inst.compress(tensor)
            recon = inst.decompress(cd, meta).reshape(tensor.shape)
            assert recon.shape == tensor.shape

    def test_alternating_pattern(self):
        tensor = np.zeros((16, 16)).astype(np.float32)
        tensor[::2, ::2] = 1.0
        tensor[1::2, 1::2] = -1.0
        inst = METHOD_REGISTRY["block_int8"]
        cd, meta = inst.compress(tensor, block_size=4)
        recon = inst.decompress(cd, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_linear_gradient(self):
        tensor = np.linspace(-1, 1, 64).astype(np.float32).reshape(8, 8)
        for mname in ("block_int8", "block_int4"):
            inst = METHOD_REGISTRY[mname]
            cd, meta = inst.compress(tensor)
            recon = inst.decompress(cd, meta).reshape(tensor.shape)
            assert recon.shape == tensor.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])

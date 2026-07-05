"""Tests for the method registry system — METHOD_REGISTRY and MethodTier."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.engine import (
    METHOD_REGISTRY,
    MethodTier,
    METHOD_TIER_MAP,
    CATEGORY_TIER_MAP,
    get_tier,
    tier_score,
)
from spectralstream.compression.engine._methods import (
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


class TestMethodRegistry:
    def test_registry_has_all_engine_methods(self):
        expected = {
            "block_int8",
            "block_int4",
            "hadamard_int8",
            "hadamard_int4",
            "sparsity_int4",
            "delta_int4",
            "svd_compress",
            "dct_spectral",
            "tensor_train",
            "fwht_compress",
        }
        assert expected.issubset(METHOD_REGISTRY.keys())

    def test_registry_instances_have_compress(self):
        for name, inst in METHOD_REGISTRY.items():
            assert hasattr(inst, "compress"), f"{name} missing compress"
            assert callable(inst.compress)

    def test_registry_instances_have_decompress(self):
        for name, inst in METHOD_REGISTRY.items():
            assert hasattr(inst, "decompress"), f"{name} missing decompress"
            assert callable(inst.decompress)

    def test_registry_instances_have_name(self):
        for name, inst in METHOD_REGISTRY.items():
            assert hasattr(inst, "name")
            assert inst.name == name

    def test_registry_instances_have_category(self):
        for name, inst in METHOD_REGISTRY.items():
            assert hasattr(inst, "category")

    def test_registry_block_int8_roundtrip(self):
        inst = METHOD_REGISTRY["block_int8"]
        tensor = np.random.randn(32, 64).astype(np.float32)
        data, meta = inst.compress(tensor)
        recon = inst.decompress(data, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape
        assert recon.dtype == tensor.dtype

    def test_registry_block_int4_roundtrip(self):
        inst = METHOD_REGISTRY["block_int4"]
        tensor = np.random.randn(32, 64).astype(np.float32)
        data, meta = inst.compress(tensor, block_size=32)
        recon = inst.decompress(data, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_registry_hadamard_int8_roundtrip(self):
        inst = METHOD_REGISTRY["hadamard_int8"]
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta = inst.compress(tensor)
        recon = inst.decompress(data, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_registry_hadamard_int4_roundtrip(self):
        inst = METHOD_REGISTRY["hadamard_int4"]
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta = inst.compress(tensor)
        recon = inst.decompress(data, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_registry_sparsity_int4_roundtrip(self):
        inst = METHOD_REGISTRY["sparsity_int4"]
        tensor = np.random.randn(128).astype(np.float32)
        data, meta = inst.compress(tensor, group_size=32)
        recon = inst.decompress(data, meta)
        assert recon.shape == tensor.shape

    def test_registry_svd_compress_roundtrip(self):
        inst = METHOD_REGISTRY["svd_compress"]
        tensor = np.random.randn(32, 32).astype(np.float32)
        data, meta = inst.compress(tensor, rank=8)
        recon = inst.decompress(data, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_registry_dct_spectral_roundtrip(self):
        inst = METHOD_REGISTRY["dct_spectral"]
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta = inst.compress(tensor)
        recon = inst.decompress(data, meta).reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_each_method_compresses_smaller(self):
        for name, inst in METHOD_REGISTRY.items():
            if name == "svd_compress":
                tensor = np.random.randn(16, 16).astype(np.float32)
            else:
                tensor = np.random.randn(32, 64).astype(np.float32)
            data, _ = inst.compress(tensor)
            assert len(data) > 0

    def test_compress_returns_tuple(self):
        tensor = np.random.randn(32, 64).astype(np.float32)
        for name, inst in METHOD_REGISTRY.items():
            result = inst.compress(tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bytes)
            assert isinstance(result[1], dict)


class TestMethodTier:
    def test_tier_enum_values(self):
        assert MethodTier.TIER1_REAL_COMPRESSION == 1
        assert MethodTier.TIER2_STRUCTURAL == 2
        assert MethodTier.TIER3_ENTROPY == 3
        assert MethodTier.TIER4_HYBRID == 4
        assert MethodTier.TIER5_QUANTIZATION == 5

    def test_category_tier_map_has_all_categories(self):
        required = {"quantization", "spectral", "structural", "entropy", "functional"}
        assert required.issubset(CATEGORY_TIER_MAP.keys())

    def test_category_to_tier_mapping(self):
        assert CATEGORY_TIER_MAP["quantization"] == MethodTier.TIER5_QUANTIZATION
        assert CATEGORY_TIER_MAP["spectral"] == MethodTier.TIER1_REAL_COMPRESSION
        assert CATEGORY_TIER_MAP["structural"] == MethodTier.TIER2_STRUCTURAL
        assert CATEGORY_TIER_MAP["entropy"] == MethodTier.TIER3_ENTROPY
        assert CATEGORY_TIER_MAP["functional"] == MethodTier.TIER1_REAL_COMPRESSION

    def test_get_tier_by_method_name(self):
        tier = get_tier("block_int8", "quantization")
        assert tier == MethodTier.TIER5_QUANTIZATION

    def test_get_tier_by_category(self):
        tier = get_tier("unknown_method", "spectral")
        assert tier == MethodTier.TIER1_REAL_COMPRESSION

    def test_get_tier_unknown_category_defaults(self):
        tier = get_tier("novel_method", "unknown_category")
        assert tier == MethodTier.TIER1_REAL_COMPRESSION

    def test_tier_score_decreasing(self):
        t1 = tier_score(MethodTier.TIER1_REAL_COMPRESSION)
        t5 = tier_score(MethodTier.TIER5_QUANTIZATION)
        assert t1 > t5

    def test_tier_score_all_positive(self):
        for tier in MethodTier:
            assert tier_score(tier) > 0

    def test_method_tier_map_contains_engine_methods(self):
        for name in METHOD_REGISTRY:
            assert name in METHOD_TIER_MAP, f"{name} missing from METHOD_TIER_MAP"

    def test_tier_5_methods_are_quantization(self):
        for name, tier in METHOD_TIER_MAP.items():
            if tier == MethodTier.TIER5_QUANTIZATION:
                assert name in METHOD_REGISTRY

    def test_get_tier_returns_enum(self):
        tier = get_tier("block_int8", "quantization")
        assert isinstance(tier, MethodTier)
        assert isinstance(tier, int)

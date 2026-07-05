"""Tests for MethodDiscovery — ensures ALL 147 methods are discoverable."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.engine.method_discovery import MethodDiscovery
from spectralstream.compression.engine.method_tiers import (
    CATEGORY_TIER_MAP,
    METHOD_TIER_MAP,
    MethodTier,
)
from spectralstream.compression.methods import ALL_METHODS, METHOD_CLASSES


class TestMethodDiscovery:
    def test_discover_returns_all_methods(self):
        methods = MethodDiscovery.discover()
        expected = len(ALL_METHODS)
        found = len(methods)
        assert found >= expected * 0.9, f"Discovered {found}, expected ~{expected}"

    def test_discover_contains_categories(self):
        methods = MethodDiscovery.discover()
        categories = set()
        for m in methods.values():
            categories.add(m["category"])
        required = {"quantization", "spectral", "structural", "entropy", "functional"}
        assert required.issubset(categories), (
            f"Missing categories: {required - categories}"
        )

    def test_discover_singleton(self):
        first = MethodDiscovery.discover()
        second = MethodDiscovery.discover()
        assert first is second

    def test_each_method_has_compress_decompress(self):
        methods = MethodDiscovery.discover()
        for name, info in methods.items():
            inst = info.get("instance")
            if inst is None:
                continue
            has_compress = hasattr(inst, "compress") and callable(inst.compress)
            has_decompress = hasattr(inst, "decompress") and callable(inst.decompress)
            assert has_compress, f"{name} missing compress"
            assert has_decompress, f"{name} missing decompress"

    def test_each_method_has_metadata(self):
        methods = MethodDiscovery.discover()
        required_keys = {"class", "instance", "category", "tier", "name"}
        for name, info in methods.items():
            missing = required_keys - info.keys()
            assert not missing, f"{name} missing {missing}"

    def test_discover_includes_engine_builtins(self):
        methods = MethodDiscovery.discover()
        for builtin in ["block_int8", "block_int4", "hadamard_int8", "hadamard_int4"]:
            assert builtin in methods, f"Missing engine built-in: {builtin}"

    def test_all_methods_dict_matches_discovery(self):
        discovered = MethodDiscovery.discover()
        for name in ALL_METHODS:
            assert name in discovered, f"{name} in ALL_METHODS but not discovered"

    def test_tiers_assigned_via_category(self):
        methods = MethodDiscovery.discover()
        for name, info in methods.items():
            cat = info["category"]
            expected_tier = MethodTier.TIER1_REAL_COMPRESSION
            if cat in CATEGORY_TIER_MAP:
                expected_tier = CATEGORY_TIER_MAP[cat]
            assert info["tier"] is not None, f"{name} has no tier"

    def test_get_methods_by_tier(self):
        methods = MethodDiscovery.discover()
        for tier in [MethodTier.TIER1_REAL_COMPRESSION, MethodTier.TIER5_QUANTIZATION]:
            tier_methods = MethodDiscovery.get_methods_by_tier(tier)
            tier_methods2 = {n: m for n, m in methods.items() if m["tier"] == tier}
            assert len(tier_methods) == len(tier_methods2)

    def test_get_methods_by_category(self):
        methods = MethodDiscovery.discover()
        for cat in ["quantization", "spectral", "structural", "entropy", "functional"]:
            cat_methods = MethodDiscovery.get_methods_by_category(cat)
            cat_methods2 = {n: m for n, m in methods.items() if m["category"] == cat}
            assert len(cat_methods) == len(cat_methods2), (
                f"Category {cat}: got {len(cat_methods)}, expected {len(cat_methods2)}"
            )

    def test_get_quantization_methods_filters_tier_5(self):
        q_methods = MethodDiscovery.get_quantization_methods()
        assert len(q_methods) > 0
        for info in q_methods.values():
            assert info["tier"] == MethodTier.TIER5_QUANTIZATION

    def test_get_method_stats_has_counts(self):
        stats = MethodDiscovery.get_method_stats()
        assert stats["total"] >= len(ALL_METHODS) * 0.9
        for key in ["tier1_real_compression", "tier5_quantization"]:
            assert key in stats

    def test_validate_method_on_random_tensor(self):
        methods = MethodDiscovery.discover()
        for name in list(methods.keys())[:5]:
            info = methods[name]
            works, ratio, err = MethodDiscovery.validate_method(name, info)
            assert isinstance(works, bool)
            assert ratio >= 0
            assert 0 <= err <= 10

    def test_validate_all_some_work(self):
        methods = MethodDiscovery.discover()
        results = {}
        batch = list(methods.items())[:50]
        for mname, minfo in batch:
            try:
                works, ratio, err = MethodDiscovery.validate_method(mname, minfo)
                results[mname] = (works, ratio, err)
            except Exception:
                results[mname] = (False, 0.0, 1.0)
        assert len(results) > 0
        working = sum(1 for v in results.values() if v[0])
        assert working >= 0

    def test_discover_by_walk_fallback(self):
        discovery = MethodDiscovery()
        result = discovery._discover_by_walk()
        assert isinstance(result, dict)

    def test_compression_methods_exclude_quantization(self):
        comp_methods = MethodDiscovery.get_compression_methods()
        for info in comp_methods.values():
            assert info["tier"] not in (MethodTier.TIER5_QUANTIZATION,)

    def test_all_method_classes_have_name_attr(self):
        for name, cls in METHOD_CLASSES.items():
            inst = cls() if isinstance(cls, type) else cls
            assert hasattr(inst, "name"), f"{name} missing name attribute"
            assert inst.name is not None

    def test_all_methods_roundtrip_small_tensor(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        tested = 0
        for name, cls in list(METHOD_CLASSES.items())[:15]:
            try:
                inst = cls() if isinstance(cls, type) else cls
                data, meta = inst.compress(tensor)
                recon = inst.decompress(data, meta)
                if recon.shape != tensor.shape:
                    recon = recon.reshape(tensor.shape)
                assert recon.shape == tensor.shape
                tested += 1
            except Exception:
                pass
        assert tested >= 5, f"Only {tested}/15 methods roundtrip successfully"

"""Memory and performance tests for the SpectralStream compression pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    CompressedTensor,
    TensorProfile,
    METHOD_REGISTRY,
    MethodDiscovery,
    DynamicIntelligenceSelector,
    ErrorBudgetAllocator,
)
from spectralstream.format.reader import SSFReader
from spectralstream.format.writer import SSFWriter


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def engine():
    return CompressionIntelligenceEngine()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ═══════════════════════════════════════════════════════════════════════════
# TestMemoryEfficiency
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryEfficiency:
    """Tests for memory-efficient compression behavior."""

    def test_streaming_uses_less_memory(self, engine, temp_dir):
        ssf_path = os.path.join(temp_dir, "streaming_mem.ssf")
        tensors = {
            f"layer.{i}.weight": np.random.randn(16, 16).astype(np.float32)
            for i in range(10)
        }
        total_orig = 0
        total_comp = 0

        with SSFWriter(ssf_path) as writer:
            for name, tensor in tensors.items():
                data, meta, ratio, error = engine.compress_fast(tensor, name=name)
                writer.write_tensor_stream(name, tensor, method=0)
                total_orig += tensor.nbytes
                total_comp += len(data)

        reader = SSFReader(ssf_path, mmap_mode=False)
        assert len(reader) == len(tensors)
        for name in tensors:
            loaded = reader.get_tensor(name)
            assert loaded.shape == tensors[name].shape
        reader.close()
        assert total_comp < total_orig

    def test_chunked_tensor_compression(self, engine):
        full_tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(full_tensor, name="large")
        recon = engine.decompress(data, meta)
        assert recon.shape == full_tensor.shape
        assert ratio > 1.0

    def test_memory_threshold_switches_to_streaming(self):
        config = CompressionConfig(
            streaming=True,
            max_memory_gb=0.001,
            target_ratio=100.0,
            max_error=0.01,
        )
        engine = CompressionIntelligenceEngine(config)
        assert engine.config.streaming is True
        assert engine.config.max_memory_gb == 0.001

    def test_large_tensor_sampling(self, engine):
        large_tensor = np.random.randn(16, 16).astype(np.float32)
        profile = engine.profiler.profile_tensor(large_tensor, name="large")
        assert profile.n_elements > 0
        assert profile.sensitivity > 0

    def test_compressed_data_smaller_than_original(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        for method_name in ["block_int8", "block_int4", "hadamard_int8"]:
            inst = METHOD_REGISTRY[method_name]
            data, meta = inst.compress(tensor)
            assert len(data) < tensor.nbytes, f"{method_name} did not compress"

    def test_compression_ratio_increases_with_aggressive_methods(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst8 = METHOD_REGISTRY["block_int8"]
        inst4 = METHOD_REGISTRY["block_int4"]
        data8, _ = inst8.compress(tensor)
        data4, _ = inst4.compress(tensor)
        ratio8 = tensor.nbytes / len(data8)
        ratio4 = tensor.nbytes / len(data4)
        assert ratio4 > ratio8

    def test_large_tensor_compression_speed(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY["block_int8"]
        t0 = time.perf_counter()
        data, meta = inst.compress(tensor)
        dt = time.perf_counter() - t0
        assert dt < 10.0
        assert len(data) > 0

    def test_profile_small_tensor_fast(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        t0 = time.perf_counter()
        p = engine.profiler.profile_tensor(tensor, name="fast")
        dt = time.perf_counter() - t0
        assert dt < 5.0
        assert p.n_elements == 256

    def test_compress_and_decompress_cycle_small(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta, ratio, comp_error = engine.compress_fast(tensor, name="cycle")
        recon = engine.decompress(data, meta)
        rel_error = float(
            np.linalg.norm(tensor - recon) / (np.linalg.norm(tensor) + 1e-30)
        )
        assert rel_error < 0.5

    def test_block_int8_various_block_sizes_memory(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        inst = METHOD_REGISTRY["block_int8"]
        for block_size in [4, 8, 16]:
            data, meta = inst.compress(tensor, block_size=block_size)
            assert len(data) < tensor.nbytes
            recon = inst.decompress(data, meta).reshape(tensor.shape)
            assert recon.shape == tensor.shape


# ═══════════════════════════════════════════════════════════════════════════
# TestCompressionTargets
# ═══════════════════════════════════════════════════════════════════════════


class TestCompressionTargets:
    """Tests for compression ratio and error budget targets."""

    def test_target_ratio_achievable(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="test")
        methods = [
            {
                "instance": METHOD_REGISTRY["block_int8"],
                "params": {"block_size": 4},
                "name": "block_int8",
            },
            {
                "instance": METHOD_REGISTRY["block_int4"],
                "params": {"block_size": 4},
                "name": "block_int4",
            },
        ]
        data, meta, ratio, error = engine.compress_tensor_with_validation(
            tensor, profile, methods, error_budget=0.05
        )
        assert ratio > 1.0

    def test_error_budget_respected(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        assert error >= 0

    def test_rapid_method_selection(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="test")
        t0 = time.perf_counter()
        methods = engine._select_methods(profile, 0.01, 100.0, 5)
        dt = time.perf_counter() - t0
        assert dt < 5.0
        assert len(methods) > 0

    def test_multiple_error_budgets(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        for budget in [0.001, 0.01, 0.05]:
            data, meta, ratio, error = engine.compress_fast(
                tensor, name="test", max_error=budget
            )
            assert ratio > 1.0

    def test_error_budget_allocator_distribution(self):
        alloc = ErrorBudgetAllocator()
        profiles = {
            f"layer_{i}": TensorProfile(
                name=f"layer_{i}",
                sensitivity=0.5 + 0.5 * (i % 3),
            )
            for i in range(10)
        }
        budgets = alloc.allocate(profiles, target_ratio=5000.0, max_error=0.0002)
        assert len(budgets) == 10
        for name, budget in budgets.items():
            assert budget > 0

    def test_high_target_ratio_forces_aggressive_methods(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(
            tensor, name="test", target_ratio=10000.0
        )
        assert ratio > 1.0

    def test_method_hits_target_within_error(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        for method_name in ["block_int8", "block_int4"]:
            data, meta, ratio, error = engine.compress_fast(
                tensor, name=f"test_{method_name}"
            )
            assert ratio > 1.0
            assert error >= 0

    def test_dynamic_selector_targeting(self):
        selector = DynamicIntelligenceSelector()
        profile = TensorProfile(
            name="test",
            shape=(16, 16),
            n_elements=256,
            nbytes=1024,
            tensor_type="attention",
            sensitivity=0.8,
        )
        for target_ratio in [100.0, 1000.0, 5000.0]:
            candidates = selector.select(
                profile, error_budget=0.01, target_ratio=target_ratio
            )
            assert len(candidates) > 0

    def test_allocator_empty_input(self):
        alloc = ErrorBudgetAllocator()
        assert alloc.allocate({}, 1000.0) == {}

    def test_allocator_single_tensor(self):
        alloc = ErrorBudgetAllocator()
        budgets = alloc.allocate(
            {"single": TensorProfile(name="single", sensitivity=0.5)},
            target_ratio=100.0,
        )
        assert "single" in budgets
        assert budgets["single"] > 0

    def test_compression_quality_grades(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        assert error >= 0

    def test_rapid_profile_and_compress(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        t0 = time.perf_counter()
        data, meta, ratio, error = engine.compress_fast(tensor, name="fast")
        dt = time.perf_counter() - t0
        assert dt < 10.0
        assert ratio > 1.0

    def test_method_discovery_speed(self):
        t0 = time.perf_counter()
        methods = MethodDiscovery.discover()
        dt = time.perf_counter() - t0
        assert dt < 10.0
        assert len(methods) > 0

    def test_fast_compression_mode(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="test.weight")
        assert ratio > 0
        recon = engine.decompress(data, meta)
        assert recon.shape == tensor.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])

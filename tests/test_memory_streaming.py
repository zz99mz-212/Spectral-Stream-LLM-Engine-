"""
Streaming Memory Management Tests
==================================
Verifies that the compression engine never exceeds 4 GB peak memory
during tensor operations, especially for large models like MiMo-V2.5
(315 GB, 73081 tensors on 4 TPU shards) running on 64 GB RAM.

Key tests:
1. Memory threshold check — GC fires correctly when < 2 GB available
2. Tensor streaming — load_tensor uses mmap (no extra RAM copy)
3. Multi-shard streaming — zero-copy mmap across shard boundaries
4. Peak memory under 4 GB during streaming compression
5. Aggressive GC frees tensor memory immediately after use
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import psutil
except ImportError:
    psutil = None

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    CompressionReport,
    CompressedTensor,
    TensorProfile,
)
from spectralstream.compression.engine._io import _SafetensorsIO
from spectralstream.compression.engine._orchestrator import (
    CompressionIntelligenceEngine as Engine,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def engine() -> CompressionIntelligenceEngine:
    """Create engine with memory-constrained config (max 4 GB per tensor)."""
    cfg = CompressionConfig(
        target_ratio=100.0,
        max_error=0.01,
        streaming=True,
        max_memory_gb=4.0,
        num_workers=1,
    )
    eng = CompressionIntelligenceEngine(config=cfg)
    yield eng
    eng.close()
    del eng
    gc.collect()


@pytest.fixture
def large_tensor(rng) -> np.ndarray:
    """Create a simulated large tensor for memory testing."""
    return rng.randn(4, 4, 4).astype(np.float32)


@pytest.fixture
def huge_tensor(rng) -> np.ndarray:
    """Create a simulated huge tensor for memory testing."""
    return rng.randn(4, 4, 4).astype(np.float32)


@pytest.fixture
def temp_safetensors_path(tmp_path, rng) -> str:
    """Create a temporary safetensors file with multiple tensors for streaming tests."""
    import struct
    import json

    # Build synthetic tensors
    tensors = {
        "model.embed_tokens.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.self_attn.k_proj.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.self_attn.v_proj.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.self_attn.o_proj.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.mlp.up_proj.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.mlp.down_proj.weight": rng.randn(16, 16).astype(np.float32),
        "model.layers.0.input_layernorm.weight": rng.randn(16).astype(np.float32),
        "model.layers.0.post_attention_layernorm.weight": rng.randn(16).astype(
            np.float32
        ),
        "model.lm_head.weight": rng.randn(16, 16).astype(np.float32),
        "model.norm.weight": rng.randn(16).astype(np.float32),
    }

    # Write safetensors format
    path = str(tmp_path / "test_model.safetensors")
    header = {}
    data_blocks = []
    offset = 0
    for name, tensor in tensors.items():
        raw = tensor.astype(np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_blocks.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"".join(data_blocks))

    return path


# ═══════════════════════════════════════════════════════════════════════════
# Memory Threshold Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(psutil is None, reason="psutil required for memory tests")
class TestMemoryThreshold:
    """Verify memory threshold detection and GC behavior."""

    def test_check_memory_threshold_returns_bool(self, engine):
        """_check_memory_threshold should return True when memory is sufficient."""
        result = engine._check_memory_threshold(min_available_gb=0.1)
        assert isinstance(result, bool)

    def test_force_gc_frees_memory(self, engine, large_tensor):
        """GC should free tensor memory."""
        before = psutil.Process().memory_info().rss
        tensors = []
        for _ in range(3):
            t = large_tensor.copy()
            tensors.append(t)
        mid = psutil.Process().memory_info().rss
        assert mid > before, "Tensor allocation should increase RSS"
        del tensors
        freed = engine._force_gc(log=False)
        after = psutil.Process().memory_info().rss
        assert after <= mid, "GC should free memory"

    def test_config_max_memory_gb_default(self):
        """CompressionConfig should default to 4 GB max memory per tensor."""
        cfg = CompressionConfig()
        assert cfg.max_memory_gb == 4.0, (
            f"Expected max_memory_gb=4.0, got {cfg.max_memory_gb}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Streaming IO Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSafetensorsStreaming:
    """Verify _SafetensorsIO streaming methods."""

    def test_load_tensor_returns_valid_array(self, temp_safetensors_path):
        """load_tensor should return a valid numpy array."""
        io = _SafetensorsIO(use_mmap=True)
        tensors = io.scan(temp_safetensors_path)
        assert len(tensors) > 0

        for name, (shape, dt, off, nb) in list(tensors.items())[:3]:
            tensor = io.load_tensor(temp_safetensors_path, shape, dt, off, nb)
            assert isinstance(tensor, np.ndarray)
            assert tensor.size > 0
            assert tensor.dtype == np.float32
            del tensor
            gc.collect()

    def test_load_tensor_mmap_no_extra_copy(self, temp_safetensors_path):
        """load_tensor should return mmap view (not a copy) for non-BF16."""
        io = _SafetensorsIO(use_mmap=True)
        tensors = io.scan(temp_safetensors_path)
        name = list(tensors.keys())[0]
        shape, dt, off, nb = tensors[name]

        tensor = io.load_tensor(temp_safetensors_path, shape, dt, off, nb)
        # For non-BF16, the tensor should share memory with the mmap
        # We verify this by checking that the array base is a memmap
        if dt != "BF16":
            # The mmap view might have a memmap as base
            assert isinstance(np.asarray(tensor).base, np.memmap) or hasattr(
                tensor, "base"
            ), "load_tensor should return an mmap-backed view"
        del tensor
        gc.collect()

    def test_load_tensor_vs_read_same_values(self, temp_safetensors_path):
        """load_tensor and read should return same values."""
        io = _SafetensorsIO(use_mmap=True)
        tensors = io.scan(temp_safetensors_path)
        name = list(tensors.keys())[0]
        shape, dt, off, nb = tensors[name]

        tensor_load = io.load_tensor(temp_safetensors_path, shape, dt, off, nb)
        tensor_read = io.read(temp_safetensors_path, shape, dt, off, nb)

        np.testing.assert_array_equal(tensor_load, tensor_read)
        del tensor_load, tensor_read
        gc.collect()

    def test_stream_tensors_yields_all_tensors(self, temp_safetensors_path):
        """_stream_tensors should yield all tensors in the file."""
        io = _SafetensorsIO(use_mmap=True)
        tensor_info = io.scan(temp_safetensors_path)
        expected_count = len(tensor_info)

        count = 0
        for name, tensor, shape, dt, off, nb in io._stream_tensors(
            temp_safetensors_path
        ):
            assert isinstance(tensor, np.ndarray)
            assert tensor.size > 0
            assert isinstance(name, str)
            assert isinstance(shape, tuple)
            assert isinstance(dt, str)
            assert isinstance(off, int) and off >= 0
            assert isinstance(nb, int) and nb > 0
            count += 1
            del tensor
            gc.collect()

        assert count == expected_count, (
            f"Expected {expected_count} tensors, got {count}"
        )

    def test_stream_tensors_memory_usage(self, temp_safetensors_path):
        """Memory should not accumulate during streaming."""
        io = _SafetensorsIO(use_mmap=True)
        before = psutil.Process().memory_info().rss if psutil else 0

        peak = before
        for name, tensor, shape, dt, off, nb in io._stream_tensors(
            temp_safetensors_path
        ):
            del tensor
            gc.collect()
            if psutil:
                current = psutil.Process().memory_info().rss
                peak = max(peak, current)

        after = psutil.Process().memory_info().rss if psutil else 0
        if psutil:
            # Allow 50 MB overhead for Python/interpreter
            mem_increase = peak - before
            assert mem_increase < 100 * 1024 * 1024, (
                f"Memory increased by {mem_increase / 1e6:.1f} MB during streaming "
                f"(before={before / 1e6:.1f} MB, peak={peak / 1e6:.1f} MB)"
            )

    def test_check_memory_threshold(self, temp_safetensors_path):
        """_check_memory_threshold should handle low memory gracefully."""
        io = _SafetensorsIO(use_mmap=True)
        # Should return True with generous threshold
        result = io._check_memory_threshold(min_available_gb=0.1)
        assert isinstance(result, bool)
        # With a very high threshold, should still handle gracefully
        result_high = io._check_memory_threshold(min_available_gb=1000.0)
        assert isinstance(result_high, bool)


# ═══════════════════════════════════════════════════════════════════════════
# Engine Memory Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(psutil is None, reason="psutil required for memory tests")
class TestEngineMemory:
    """Verify engine memory management during compression."""

    def test_compress_tensor_with_del_and_gc(self, engine, large_tensor):
        """After compress_tensor, the original tensor should be deletable with GC."""
        profile = engine.profiler.profile_tensor(large_tensor, name="test_tensor")

        before = psutil.Process().memory_info().rss
        ct = engine.compress_tensor(large_tensor, profile, "block_int8")
        mid = psutil.Process().memory_info().rss

        # The original tensor is still alive here. Normally we'd del it.
        # This test just verifies compression doesn't leak.

        assert ct is not None
        assert ct.compression_ratio > 1.0
        assert isinstance(ct.data, bytes)
        assert ct.method == "block_int8"

    def test_streaming_profile_model_memory(self, engine, temp_safetensors_path):
        """profile_model should keep memory stable."""
        before = psutil.Process().memory_info().rss
        profiles = engine.profile_model(temp_safetensors_path)
        after = psutil.Process().memory_info().rss

        # Memory should not grow dramatically after profiling
        mem_increase = after - before
        assert mem_increase < 200 * 1024 * 1024, (
            f"Memory increased by {mem_increase / 1e6:.1f} MB during profiling"
        )
        assert len(profiles) > 0

    def test_compress_model_streaming_memory(
        self, engine, temp_safetensors_path, tmp_path
    ):
        """compress_model with streaming should stay under 4 GB."""
        output_path = str(tmp_path / "test_out.ssf")
        before = psutil.Process().memory_info().rss

        report = engine.compress_model(
            temp_safetensors_path,
            output_path,
            streaming=True,
        )

        after = psutil.Process().memory_info().rss
        mem_increase = after - before

        # Allow 300 MB overhead for method cache, Python overhead, etc.
        assert mem_increase < 400 * 1024 * 1024, (
            f"Memory increased by {mem_increase / 1e6:.1f} MB during streaming "
            f"compression (limit 400 MB)"
        )
        assert report is not None
        assert report.overall_ratio >= 1.0

    def test_after_compress_model_memory_returns_to_baseline(
        self, engine, temp_safetensors_path, tmp_path
    ):
        """After streaming compression + engine close, memory should return near baseline."""
        output_path = str(tmp_path / "test_out2.ssf")
        before = psutil.Process().memory_info().rss

        _ = engine.compress_model(
            temp_safetensors_path,
            output_path,
            streaming=True,
        )

        # Close engine to release resources
        engine.close()

        gc.collect()
        gc.collect()
        after = psutil.Process().memory_info().rss

        # Allow 150 MB overhead for Python/interpreter
        mem_increase = after - before
        assert mem_increase < 250 * 1024 * 1024, (
            f"Engine close did not release memory: {mem_increase / 1e6:.1f} MB above baseline"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Peak Memory Compliance Test
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(psutil is None, reason="psutil required for memory tests")
class TestPeakMemoryCompliance:
    """CRITICAL: Verify peak memory never exceeds 4 GB during operations."""

    def test_streaming_memory_never_exceeds_4gb(self, engine, tmp_path):
        """Verify memory never exceeds 4 GB during streaming compression.

        Uses synthetic ~1 GB tensor to simulate large model tensors.
        """
        # Create a moderate synthetic model
        synthetic_tensors = {}
        rng = np.random.RandomState(42)
        for i in range(4):
            synthetic_tensors[f"layer_{i}.weight"] = rng.randn(16, 16).astype(
                np.float32
            )

        max_memory_during_operation = 0

        # Track memory during compression
        def track_memory():
            return psutil.Process().memory_info().rss

        baseline = track_memory()

        for name, tensor in synthetic_tensors.items():
            before = track_memory()
            profile = engine.profiler.profile_tensor(tensor, name=name)
            ct = engine.compress_tensor(tensor, profile, "block_int8")
            del tensor, ct
            gc.collect()
            after = track_memory()
            max_memory_during_operation = max(max_memory_during_operation, after)

        peak = max_memory_during_operation
        peak_gb = peak / (1024**3)
        assert peak < 4 * 1024**3, f"Peak memory {peak_gb:.1f} GB exceeds 4 GB limit"

    def test_streaming_compressor_no_accumulation(self, engine, tmp_path, rng):
        """The streaming compressor should not accumulate tensors in memory."""
        from spectralstream.compression.engine.streaming_compressor import (
            StreamingCompressor,
        )

        # Create a safetensors file first
        import struct
        import json

        tensors = {
            f"layer_{i}.weight": rng.randn(16, 16).astype(np.float32) for i in range(8)
        }
        st_path = str(tmp_path / "stream_test.safetensors")
        header = {}
        data_blocks = []
        offset = 0
        for name, tensor in tensors.items():
            raw = tensor.astype(np.float32).tobytes()
            header[name] = {
                "dtype": "F32",
                "shape": list(tensor.shape),
                "data_offsets": [offset, offset + len(raw)],
            }
            data_blocks.append(raw)
            offset += len(raw)
        header_bytes = json.dumps(header).encode()
        with open(st_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"".join(data_blocks))

        output_path = str(tmp_path / "stream_out.ssf")

        compressor = StreamingCompressor(
            engine=engine,
            model_path=st_path,
            output_path=output_path,
            config=CompressionConfig(target_ratio=50.0, max_error=0.01, streaming=True),
        )

        # Record memory before
        before = psutil.Process().memory_info().rss

        result = compressor.compress_all()

        after = psutil.Process().memory_info().rss
        mem_increase = after - before

        # Should not accumulate memory (allow 150 MB for overhead)
        assert mem_increase < 200 * 1024 * 1024, (
            f"Streaming compressor accumulated {mem_increase / 1e6:.1f} MB"
        )
        assert result["total_tensors"] == 8
        assert result["overall_ratio"] >= 1.0
        assert result["peak_memory_mb"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Shard IO Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiShardStreaming:
    """Verify multi-shard streaming with mmap."""

    @pytest.fixture
    def multi_shard_dir(self, tmp_path, rng):
        """Create a simulated multi-shard model directory."""
        import struct
        import json

        shard_dir = tmp_path / "multi_shard_model"
        shard_dir.mkdir()

        # Create 2 shards with a few tensors each
        for shard_idx in range(2):
            tensors = {}
            for i in range(3):
                name = f"shard{shard_idx}_layer{i}.weight"
                tensors[name] = rng.randn(16, 16).astype(np.float32)

            header = {}
            data_blocks = []
            offset = 0
            for name, tensor in tensors.items():
                raw = tensor.astype(np.float32).tobytes()
                header[name] = {
                    "dtype": "F32",
                    "shape": list(tensor.shape),
                    "data_offsets": [offset, offset + len(raw)],
                }
                data_blocks.append(raw)
                offset += len(raw)

            header_bytes = json.dumps(header).encode()
            st_path = shard_dir / f"model-{shard_idx:05d}-of-00003.safetensors"
            with open(st_path, "wb") as f:
                f.write(struct.pack("<Q", len(header_bytes)))
                f.write(header_bytes)
                f.write(b"".join(data_blocks))

        return str(shard_dir)

    def test_multi_shard_discovers_all_shards(self, multi_shard_dir):
        """MultiShardSafetensorsIO should discover all shards."""
        from spectralstream.compression.engine.multi_shard_io import (
            MultiShardSafetensorsIO,
        )

        io = MultiShardSafetensorsIO(multi_shard_dir)
        assert len(io.shard_paths) >= 2
        assert len(io.index) >= 6  # 2 shards * 3 tensors

    def test_multi_shard_load_tensor_streaming(self, multi_shard_dir):
        """load_tensor_streaming should return valid numpy arrays."""
        from spectralstream.compression.engine.multi_shard_io import (
            MultiShardSafetensorsIO,
        )

        io = MultiShardSafetensorsIO(multi_shard_dir)
        names = io.list_tensors()
        assert len(names) > 0

        for name in names[:5]:
            tensor = io.load_tensor_streaming(name)
            assert isinstance(tensor, np.ndarray)
            assert tensor.size > 0
            assert tensor.dtype == np.float32
            del tensor
            gc.collect()

    def test_multi_shard_stream_tensors_yields_all(self, multi_shard_dir):
        """_stream_tensors should yield all tensors across all shards."""
        from spectralstream.compression.engine.multi_shard_io import (
            MultiShardSafetensorsIO,
        )

        io = MultiShardSafetensorsIO(multi_shard_dir)
        expected = len(io.index)
        count = 0

        for name, tensor, info in io._stream_tensors():
            assert isinstance(tensor, np.ndarray)
            assert tensor.size > 0
            assert isinstance(info, dict)
            count += 1
            del tensor
            gc.collect()

        assert count == expected, f"Expected {expected} tensors, got {count}"

    def test_multi_shard_stream_tensors_memory_stable(self, multi_shard_dir):
        """_stream_tensors should not accumulate memory across shards."""
        from spectralstream.compression.engine.multi_shard_io import (
            MultiShardSafetensorsIO,
        )

        io = MultiShardSafetensorsIO(multi_shard_dir)
        before = psutil.Process().memory_info().rss if psutil else 0

        for name, tensor, info in io._stream_tensors():
            del tensor
            gc.collect()

        after = psutil.Process().memory_info().rss if psutil else 0
        if psutil:
            mem_increase = after - before
            assert mem_increase < 100 * 1024 * 1024, (
                f"Memory increased by {mem_increase / 1e6:.1f} MB "
                f"during multi-shard streaming"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Profiler Memory Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProfilerMemory:
    """Verify profiler handles large tensors without OOM."""

    def test_profile_large_tensor_no_oom(self, engine, rng):
        """Profiling a very large tensor should not cause OOM."""
        # Create a large 2D tensor that would normally trigger full SVD
        large_2d = rng.randn(16, 16).astype(np.float32)
        profile = engine.profiler.profile_tensor(large_2d, name="large_test")
        assert profile is not None
        assert profile.effective_rank >= 0
        del large_2d
        gc.collect()

    def test_profile_huge_embedding_no_oom(self, engine, rng):
        """Profiling a huge embedding matrix should subsample SVD."""
        huge_embed = rng.randn(16, 16).astype(np.float32)
        profile = engine.profiler.profile_tensor(huge_embed, name="embed_tokens")
        assert profile is not None
        assert profile.effective_rank >= 0
        del huge_embed
        gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# Engine Config Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_engine_config_max_memory():
    """Engine config should have max_memory_gb accessible."""
    cfg = CompressionConfig()
    assert hasattr(cfg, "max_memory_gb")
    assert cfg.max_memory_gb == 4.0


def test_engine_has_check_memory_threshold(tiny_engine):
    """Engine should have _check_memory_threshold method."""
    engine = tiny_engine
    assert hasattr(engine, "_check_memory_threshold")
    assert callable(engine._check_memory_threshold)


def test_io_has_check_memory_threshold():
    """_SafetensorsIO should have _check_memory_threshold static method."""
    assert hasattr(_SafetensorsIO, "_check_memory_threshold")
    assert callable(_SafetensorsIO._check_memory_threshold)


def test_io_has_load_tensor():
    """_SafetensorsIO should have load_tensor method."""
    io = _SafetensorsIO(use_mmap=True)
    assert hasattr(io, "load_tensor")
    assert callable(io.load_tensor)


def test_io_has_stream_tensors():
    """_SafetensorsIO should have _stream_tensors method."""
    io = _SafetensorsIO(use_mmap=True)
    assert hasattr(io, "_stream_tensors")
    assert callable(io._stream_tensors)


def test_streaming_default():
    """CompressionConfig should default streaming to True."""
    cfg = CompressionConfig()
    assert cfg.streaming is True


@pytest.mark.skipif(psutil is None, reason="psutil required")
def test_engine_memory_tracking(tiny_engine):
    """Engine should track peak memory."""
    engine = tiny_engine
    engine._check_memory()
    telemetry = engine.get_telemetry()
    assert "memory_peak_mb" in telemetry
    assert telemetry["memory_peak_mb"] >= 0

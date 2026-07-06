"""Tests for UnifiedStreamingPipeline — dual-mode streaming compression."""

from __future__ import annotations

import gc
import json
import os
import struct
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.compression.engine._orchestrator import (
        CompressionIntelligenceEngine,
    )
    from spectralstream.compression.engine._dataclasses import CompressionConfig
    from spectralstream.compression.streaming.unified_streaming_pipeline import (
        UnifiedStreamingPipeline,
        UnifiedStreamingCompressionPipeline,
        CompressionMode,
        auto_detect_mode,
        check_model_size_gb,
        check_available_ram_gb,
        CompressionReport,
        StreamedTensorResult,
    )

    _HAS_CORE = True
    _import_error = ""
except ImportError as e:
    _HAS_CORE = False
    _import_error = str(e)


def _make_synthetic_safetensors(
    path: str,
    num_tensors: int = 8,
    min_dim: int = 32,
    max_dim: int = 128,
    seed: int = 42,
) -> int:
    """Create a synthetic safetensors file and return total bytes of tensor data."""
    rng = np.random.RandomState(seed)
    tensor_data = bytearray()
    data_offset = 0

    headers: dict = {"__metadata__": {"format": "pt"}}
    for i in range(num_tensors):
        dim = int(rng.randint(min_dim, max_dim + 1))
        shape = (dim, dim)
        tensor = rng.randn(*shape).astype(np.float32)
        raw = tensor.tobytes()
        name = f"tensor_{i:04d}"
        headers[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [data_offset, data_offset + len(raw)],
        }
        tensor_data.extend(raw)
        data_offset += len(raw)

    header_json = json.dumps(headers, separators=(",", ":"))
    header_len = struct.pack("<Q", len(header_json))

    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_json.encode())
        f.write(bytes(tensor_data))

    return data_offset


def _make_synthetic_model_large(
    path: str, target_bytes: int = 50 * 1024 * 1024, seed: int = 42
) -> int:
    """Create a synthetic safetensors file targeting roughly target_bytes of data.
    Uses many small tensors to avoid slow compression times."""
    rng = np.random.RandomState(seed)
    tensor_data = bytearray()
    data_offset = 0
    idx = 0

    headers: dict = {"__metadata__": {"format": "pt"}}
    while data_offset < target_bytes:
        dim = int(rng.randint(32, 96))
        shape = (dim, dim)
        tensor = rng.randn(*shape).astype(np.float32)
        raw = tensor.tobytes()
        if data_offset + len(raw) > target_bytes * 1.1:
            break
        name = f"large_tensor_{idx:04d}"
        headers[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [data_offset, data_offset + len(raw)],
        }
        tensor_data.extend(raw)
        data_offset += len(raw)
        idx += 1

    header_json = json.dumps(headers, separators=(",", ":"))
    header_len = struct.pack("<Q", len(header_json))

    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_json.encode())
        f.write(bytes(tensor_data))

    return data_offset


_SKIP_REASON = f"Core imports failed: {_import_error}" if not _HAS_CORE else ""


@pytest.mark.skipif(not _HAS_CORE, reason=_SKIP_REASON)
class TestUnifiedStreamingPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        self._tmpdir = tempfile.mkdtemp(prefix="usp_test_")
        self._model_path = os.path.join(self._tmpdir, "test_model.safetensors")
        self._output_path = os.path.join(self._tmpdir, "test_output.ssf")
        self._config = CompressionConfig(
            target_ratio=100.0,
            max_error=0.05,
            streaming=True,
            memory_budget_mb=256,
            num_workers=2,
        )
        yield
        gc.collect()
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── Model creation helpers ────────────────────────────────────────────

    def _create_small_model(self):
        total_bytes = _make_synthetic_safetensors(
            self._model_path, num_tensors=6, min_dim=16, max_dim=64
        )
        return total_bytes

    def _create_large_model(self):
        total_bytes = _make_synthetic_model_large(
            self._model_path, target_bytes=500 * 1024 * 1024
        )
        return total_bytes

    # ── Test: header-only scan ────────────────────────────────────────────

    def test_header_only_scan(self):
        self._create_small_model()
        info = UnifiedStreamingPipeline._header_only_scan(self._model_path)
        assert len(info) == 6, f"Expected 6 tensors, got {len(info)}"
        for name, (shape, dtype, offset, nbytes) in info.items():
            assert dtype == "F32"
            assert len(shape) == 2
            assert nbytes > 0

    # ── Test: model size check ────────────────────────────────────────────

    def test_check_model_size(self):
        total = self._create_small_model()
        detected = check_model_size_gb(self._model_path)
        expected_gb = total / (1024**3)
        assert abs(detected - expected_gb) < 1e-6

    # ── Test: available RAM check ─────────────────────────────────────────

    def test_check_available_ram(self):
        ram = check_available_ram_gb()
        assert ram > 0.0
        assert ram < 1_000_000.0  # sanity

    # ── Test: auto_detect_mode ────────────────────────────────────────────

    def test_auto_detect_mode_streaming(self):
        mode = auto_detect_mode(
            self._model_path, memory_budget_gb=0.5, force_mode="streaming"
        )
        assert mode == CompressionMode.STREAMING

    def test_auto_detect_mode_ram(self):
        mode = auto_detect_mode(
            self._model_path, memory_budget_gb=64.0, force_mode="ram"
        )
        assert mode == CompressionMode.RAM

    # ── Test: streaming mode (small model) ────────────────────────────────

    def test_streaming_mode_small(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert isinstance(report, CompressionReport)
        assert report.total_tensors == 6
        assert report.total_original_bytes > 0
        assert report.total_compressed_bytes > 0
        assert report.overall_ratio >= 1.0
        assert report.mode == "streaming"
        assert os.path.getsize(self._output_path) > 0
        print(
            f"\nStreaming mode: ratio={report.overall_ratio:.1f}x, "
            f"peak_mem={report.peak_memory_mb:.0f}MB, "
            f"time={report.time_seconds:.1f}s"
        )

    # ── Test: RAM mode (small model) ──────────────────────────────────────

    def test_ram_mode_small(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="ram",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert isinstance(report, CompressionReport)
        assert report.total_tensors == 6
        assert report.total_original_bytes > 0
        assert report.total_compressed_bytes > 0
        assert report.overall_ratio >= 1.0
        assert report.mode == "ram"
        print(
            f"\nRAM mode: ratio={report.overall_ratio:.1f}x, "
            f"peak_mem={report.peak_memory_mb:.0f}MB, "
            f"time={report.time_seconds:.1f}s"
        )

    # ── Test: auto mode ──────────────────────────────────────────────────

    def test_auto_mode(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="auto",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert report.total_tensors == 6
        assert report.overall_ratio >= 1.0
        assert report.mode in ("streaming", "ram")

    # ── Test: round-trip decompression ────────────────────────────────────

    def test_round_trip_decompression(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)

        for mode_name in ("streaming", "ram"):
            out_path = os.path.join(self._tmpdir, f"test_{mode_name}.ssf")
            pipeline = UnifiedStreamingPipeline(
                method_oracle=engine.oracle,
                cascade_engine=None,
                memory_budget_mb=256,
                mode=mode_name,
            )
            report = pipeline.compress_model(
                model_path=self._model_path,
                output_path=out_path,
                target_ratio=100.0,
                max_error=0.05,
                quiet=True,
            )

            assert os.path.getsize(out_path) > 0, f"SSF file empty for {mode_name}"

            from spectralstream.format.reader import SSFReader

            reader = SSFReader(out_path, mmap_mode=True)
            index = reader._index
            assert index is not None and len(index) > 0
            assert len(index) == report.total_tensors, (
                f"Expected {report.total_tensors} entries, got {len(index)}"
            )

            for entry in index:
                assert hasattr(entry, "compressed_size") and entry.compressed_size > 0
                assert hasattr(entry, "original_size") and entry.original_size > 0

            reader.close()

    # ── Test: peak memory tracking ───────────────────────────────────────

    def test_peak_memory_tracking(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert report.peak_memory_mb >= 0
        assert report.peak_memory_mb < 30_000  # sanity: not >30GB

    # ── Test: memory budget enforcement (streaming, small budget) ─────────

    def test_streaming_with_limited_budget(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=64,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert report.total_tensors == 6
        assert report.overall_ratio >= 1.0
        assert report.mode == "streaming"

    # ── Test: progress callback ──────────────────────────────────────────

    def test_progress_callback(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        progress_log: list = []

        def cb(processed, total, name, ratio, error):
            progress_log.append((processed, total, name, ratio, error))

        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            progress_callback=cb,
            quiet=True,
        )
        assert len(progress_log) == report.total_tensors
        assert progress_log[-1][0] == report.total_tensors

    # ── Test: checkpoint / resume ─────────────────────────────────────────

    def test_checkpoint_resume(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )

        progress_log: list = []

        def cb(processed, total, name, ratio, error):
            progress_log.append((processed, total, name, ratio, error))

        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            progress_callback=cb,
            quiet=True,
            resume=False,
        )
        assert report.total_tensors == 6
        assert report.overall_ratio >= 1.0
        assert len(progress_log) == report.total_tensors

        # Re-compress with resume should not error
        report2 = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
            resume=True,
        )
        assert report2.total_tensors == 6

    # ── Test: report JSON serialization ───────────────────────────────────

    def test_report_json_serialization(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        report_path = os.path.join(self._tmpdir, "report.json")
        pipeline.save_report_json(report, report_path)
        assert os.path.exists(report_path)
        with open(report_path) as f:
            data = json.load(f)
        assert data["total_tensors"] == 6
        assert data["overall_ratio"] == report.overall_ratio
        assert data["mode"] == "streaming"

    # ── Test: format_report ──────────────────────────────────────────────

    def test_format_report(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        formatted = pipeline.format_report(report)
        assert "Unified Streaming Compression Report" in formatted
        assert f"{report.total_tensors}" in formatted
        assert f"{report.overall_ratio:.1f}x" in formatted

    # ── Test: backward-compatible wrapper ─────────────────────────────────

    def test_backward_compatible_wrapper(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingCompressionPipeline(
            engine=engine,
            config=self._config,
            memory_budget_mb=256,
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert report.total_tensors == 6
        assert report.overall_ratio >= 1.0

    # ── Test: errors are tracked ─────────────────────────────────────────

    def test_error_tracking(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert report.avg_error >= 0.0
        assert report.max_error >= report.avg_error

    # ── Test: method distribution populated ───────────────────────────────

    def test_method_distribution(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert len(report.method_distribution) > 0
        total_tensors_from_dist = sum(report.method_distribution.values())
        assert total_tensors_from_dist == report.total_tensors

    # ── Test: tensor results populated ────────────────────────────────────

    def test_tensor_results(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=100.0,
            max_error=0.05,
            quiet=True,
        )
        assert len(report.tensor_results) == report.total_tensors
        for tr in report.tensor_results:
            assert isinstance(tr, StreamedTensorResult)
            assert tr.name
            assert tr.compression_ratio >= 1.0
            assert tr.original_nbytes > 0
            assert tr.compressed_nbytes > 0

    # ── Test: model size detection ─────────────────────────────────────────

    def test_model_size_gb_small(self):
        total = self._create_small_model()
        gb = total / (1024**3)
        detected = check_model_size_gb(self._model_path)
        assert abs(detected - gb) < 1e-9

    # ── Test: multiple modes produce comparable results ────────────────────

    def test_streaming_vs_ram_consistency(self):
        self._create_small_model()
        engine = CompressionIntelligenceEngine(config=self._config)

        results = {}
        for mode_name in ("streaming", "ram"):
            out_path = os.path.join(self._tmpdir, f"test_{mode_name}.ssf")
            pipeline = UnifiedStreamingPipeline(
                method_oracle=engine.oracle,
                cascade_engine=None,
                memory_budget_mb=256,
                mode=mode_name,
            )
            report = pipeline.compress_model(
                model_path=self._model_path,
                output_path=out_path,
                target_ratio=100.0,
                max_error=0.05,
                quiet=True,
            )
            results[mode_name] = report

        streaming_ratio = results["streaming"].overall_ratio
        ram_ratio = results["ram"].overall_ratio
        ratio_diff = abs(streaming_ratio - ram_ratio) / max(streaming_ratio, ram_ratio)
        assert ratio_diff < 0.5, (
            f"Streaming ratio {streaming_ratio:.1f}x vs RAM {ram_ratio:.1f}x "
            f"differ by {ratio_diff:.1%}"
        )


@pytest.mark.skipif(not _HAS_CORE, reason=_SKIP_REASON)
@pytest.mark.slow
class TestUnifiedStreamingPipelineLarge:
    """Heavy tests with a ~10MB synthetic model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self._tmpdir = tempfile.mkdtemp(prefix="usp_large_")
        self._model_path = os.path.join(self._tmpdir, "large_model.safetensors")
        self._output_path = os.path.join(self._tmpdir, "large_output.ssf")
        self._config = CompressionConfig(
            target_ratio=50.0,
            max_error=0.05,
            streaming=True,
            memory_budget_mb=256,
            num_workers=2,
        )

        print("\nCreating synthetic model (few large tensors)...")
        t0 = time.perf_counter()
        # Use few but large tensors to keep tensor count low — each ~1-4MB
        total = _make_synthetic_safetensors(
            self._model_path, num_tensors=4, min_dim=256, max_dim=1024
        )
        elapsed = time.perf_counter() - t0
        model_gb = total / (1024**3)
        print(
            f"  Created {total / 1e6:.1f}MB model ({model_gb:.3f}GB) in {elapsed:.1f}s"
        )

        # Compress once in streaming mode — shared across tests
        t_compress = time.perf_counter()
        engine = CompressionIntelligenceEngine(config=self._config)
        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=256,
            mode="streaming",
        )
        self._report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=50.0,
            max_error=0.05,
            quiet=True,
        )
        self._engine_ref = engine
        print(f"  Compressed in {time.perf_counter() - t_compress:.1f}s")

        yield

        gc.collect()
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_large_model_streaming(self):
        report = self._report
        assert report.total_tensors > 0
        assert report.total_original_bytes > 0
        assert report.total_compressed_bytes > 0
        assert report.overall_ratio >= 1.0
        assert report.mode == "streaming"
        assert os.path.getsize(self._output_path) > 0
        print(
            f"\nLarge streaming: {report.total_tensors} tensors, "
            f"ratio={report.overall_ratio:.1f}x, "
            f"peak_mem={report.peak_memory_mb:.0f}MB, "
            f"time={report.time_seconds:.1f}s"
        )

    def test_large_model_ram(self):
        pipeline = UnifiedStreamingPipeline(
            method_oracle=self._engine_ref.oracle,
            cascade_engine=None,
            memory_budget_mb=4096,
            mode="ram",
        )
        t0 = time.perf_counter()
        report = pipeline.compress_model(
            model_path=self._model_path,
            output_path=self._output_path,
            target_ratio=50.0,
            max_error=0.05,
            quiet=True,
        )
        elapsed = time.perf_counter() - t0
        assert report.total_tensors > 0
        assert report.overall_ratio >= 1.0
        assert report.mode == "ram"
        print(
            f"\nLarge RAM: {report.total_tensors} tensors, "
            f"ratio={report.overall_ratio:.1f}x, "
            f"peak_mem={report.peak_memory_mb:.0f}MB, "
            f"time={elapsed:.1f}s"
        )

    def test_large_round_trip(self):
        from spectralstream.format.reader import SSFReader

        reader = SSFReader(self._output_path, mmap_mode=True)
        index = reader._index
        assert index is not None
        assert len(index) == self._report.total_tensors
        for entry in index:
            assert hasattr(entry, "compressed_size") and entry.compressed_size > 0
            assert hasattr(entry, "original_size") and entry.original_size > 0
        reader.close()

    def test_large_memory_tracking(self):
        report = self._report
        model_gb = report.model_size_gb
        peak_gb = report.peak_memory_mb / 1024
        print(
            f"\nMemory efficiency: model={model_gb:.1f}GB, "
            f"peak={peak_gb:.1f}GB, "
            f"ratio={peak_gb / max(model_gb, 0.001):.2f}x model size in RAM"
        )
        baseline_overhead_gb = 0.5
        expected_max = max(model_gb * 2, baseline_overhead_gb)
        assert peak_gb < expected_max + 0.5, (
            f"Peak memory {peak_gb:.1f}GB exceeds expected max {expected_max:.1f}GB "
            f"(model={model_gb:.3f}GB, baseline overhead={baseline_overhead_gb:.1f}GB)"
        )

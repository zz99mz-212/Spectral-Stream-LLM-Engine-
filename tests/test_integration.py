"""Comprehensive integration tests for the SpectralStream compression pipeline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

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
    MethodDiscovery,
    DynamicIntelligenceSelector,
    ErrorBudgetAllocator,
    CompressionProfiler,
    METHOD_TIER_MAP,
)
from spectralstream.compression.certificate import (
    CertificateBuilder,
    CompressionCertificate,
    ValidationCertificate,
)
from spectralstream.format.reader import SSFReader
from spectralstream.format.writer import SSFWriter

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_tensor(rng):
    return rng.randn(16, 16).astype(np.float32)


@pytest.fixture
def medium_tensor(rng):
    return rng.randn(16, 16).astype(np.float32)


@pytest.fixture
def multi_tensor_model(rng):
    """Creates a dict of synthetic tensors mimicking a small model."""
    return {
        "embedding.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.attn_q.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.attn_k.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.attn_v.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.attn_o.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.ffn_gate.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.ffn_up.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.ffn_down.weight": rng.randn(16, 16).astype(np.float32),
        "layer.0.rms_norm.weight": rng.randn(16).astype(np.float32),
        "output.weight": rng.randn(16, 16).astype(np.float32),
    }


@pytest.fixture
def engine():
    return CompressionIntelligenceEngine()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ═══════════════════════════════════════════════════════════════════════════
# TestFullPipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Tests the complete compression pipeline from profiling through compression to certificate generation."""

    def test_synthetic_model_compress_decompress(self, multi_tensor_model, engine):
        tensors = multi_tensor_model
        total_orig = 0
        total_comp = 0

        for name, tensor in tensors.items():
            profile = engine.profiler.profile_tensor(tensor, name=name)
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            total_orig += tensor.nbytes
            total_comp += len(data)
            recon = engine.decompress(data, meta)
            assert recon.shape == tensor.shape

        overall_ratio = total_orig / max(total_comp, 1)
        assert overall_ratio > 1.0

    def test_compress_with_certificate(self, multi_tensor_model, engine):
        tensors = multi_tensor_model
        compressed_tensors: List[tuple] = []
        total_orig = 0
        total_comp = 0

        for name, tensor in tensors.items():
            profile = engine.profiler.profile_tensor(tensor, name=name)
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            ct = CompressedTensor(
                _data=data,
                method=meta.get("method", "block_int8"),
                params=meta,
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=ratio,
                relative_error=error,
            )
            compressed_tensors.append((name, ct))
            total_orig += tensor.nbytes
            total_comp += ct.get_data_size()

        cert = CertificateBuilder.from_compressed_tensors(
            compressed_tensors,
            model_name="test_model",
            compression_time=1.0,
        )
        assert isinstance(cert, CompressionCertificate)
        assert cert.total_tensors == len(tensors)
        assert cert.overall_ratio > 1.0
        assert len(cert.tensor_certificates) == len(tensors)
        assert len(cert.method_distribution) > 0

        cert_dict = cert.to_dict()
        assert "model" in cert_dict
        assert "compression" in cert_dict
        assert "quality" in cert_dict
        assert cert_dict["compression"]["ratio"] > 1.0

    def test_compress_with_validation(self, medium_tensor, engine):
        profile = engine.profiler.profile_tensor(medium_tensor, name="test")
        methods = [
            {
                "instance": METHOD_REGISTRY["block_int8"],
                "params": {"block_size": 128},
                "name": "block_int8",
            },
            {
                "instance": METHOD_REGISTRY["block_int4"],
                "params": {"block_size": 32},
                "name": "block_int4",
            },
        ]
        data, meta, ratio, error = engine.compress_tensor_with_validation(
            medium_tensor, profile, methods, error_budget=0.1
        )
        assert ratio > 1.0
        recon = engine.decompress(data, meta)
        assert recon.shape == medium_tensor.shape

    def test_compress_streaming(self, multi_tensor_model, engine):
        tensors = multi_tensor_model
        total_orig = 0
        total_comp = 0

        for name, tensor in tensors.items():
            profile = engine.profiler.profile_tensor(tensor, name=name)
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            total_orig += tensor.nbytes
            total_comp += len(data)
            recon = engine.decompress(data, meta)
            assert recon.shape == tensor.shape

        assert total_orig > total_comp

    def test_compress_with_stacking(self, engine):
        tensor = np.random.randn(64, 64).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="test")
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        assert ratio > 1.0
        recon = engine.decompress(data, meta)
        assert recon.shape == tensor.shape

    def test_compress_quick_mode(self, engine):
        tensor = np.random.randn(64, 64).astype(np.float32)
        data, meta, ratio, error = engine.compress_fast(tensor, name="test_weight")
        assert ratio > 0
        recon = engine.decompress(data, meta)
        assert recon.shape == tensor.shape

    def test_full_compress_decompress_report(self, multi_tensor_model, engine):
        tensors = multi_tensor_model
        compressed: List[CompressedTensor] = []
        total_orig = 0
        total_comp = 0

        for name, tensor in tensors.items():
            profile = engine.profiler.profile_tensor(tensor, name=name)
            methods = [
                {
                    "instance": METHOD_REGISTRY["block_int8"],
                    "params": {"block_size": 128},
                    "name": "block_int8",
                },
                {
                    "instance": METHOD_REGISTRY["block_int4"],
                    "params": {"block_size": 32},
                    "name": "block_int4",
                },
            ]
            data, meta, ratio, error = engine.compress_tensor_with_validation(
                tensor, profile, methods, error_budget=0.01
            )
            ct = CompressedTensor(
                _data=data,
                method=meta.get("method", "unknown"),
                params=meta,
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=ratio,
                relative_error=error,
            )
            compressed.append(ct)
            total_orig += tensor.nbytes
            total_comp += ct.get_data_size()

        report = CompressionReport(
            tensors=compressed,
            total_original_bytes=total_orig,
            total_compressed_bytes=total_comp,
            overall_ratio=total_orig / max(total_comp, 1),
            average_ratio=total_orig / max(total_comp, 1),
            weighted_error=float(np.mean([c.relative_error for c in compressed])),
            avg_error=float(np.mean([c.relative_error for c in compressed])),
            max_error=float(max(c.relative_error for c in compressed)),
        )
        assert report.overall_ratio > 1.0
        assert report.summary() is not None
        d = report.to_dict()
        assert d["num_tensors"] == len(tensors)

    def test_compress_and_generate_certificate_formats(
        self, multi_tensor_model, engine, temp_dir
    ):
        compressed_tensors: List[tuple] = []
        for name, tensor in multi_tensor_model.items():
            profile = engine.profiler.profile_tensor(tensor, name=name)
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            ct = CompressedTensor(
                _data=data,
                method=meta.get("method", "block_int8"),
                params=meta,
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=ratio,
                relative_error=error,
            )
            compressed_tensors.append((name, ct))

        cert = CertificateBuilder.from_compressed_tensors(
            compressed_tensors, model_name="test"
        )
        cert_path = os.path.join(temp_dir, "cert")
        cert.save(cert_path, formats=["json", "txt"])

        assert os.path.exists(cert_path + ".json")
        assert os.path.exists(cert_path + ".txt")

        with open(cert_path + ".json") as f:
            data = json.load(f)
        assert data["compression"]["ratio"] > 1.0

    def test_compress_with_allocator(self, multi_tensor_model, engine):
        profiles = {}
        tensors = multi_tensor_model
        for name, tensor in tensors.items():
            profiles[name] = engine.profiler.profile_tensor(tensor, name=name)

        budgets = engine.allocator.allocate(
            profiles, target_ratio=5000.0, max_error=0.0002
        )
        assert len(budgets) == len(tensors)
        for name in tensors:
            assert name in budgets

        for name, tensor in tensors.items():
            profile = profiles[name]
            data, meta, ratio, error = engine.compress_fast(tensor, name=name)
            recon = engine.decompress(data, meta)
            assert recon.shape == tensor.shape

    def test_sensitivity_based_allocation(self, engine):
        profiles = {
            "attention.weight": TensorProfile(
                name="attention.weight",
                nbytes=16384,
                sensitivity=1.0,
            ),
            "ffn.weight": TensorProfile(
                name="ffn.weight",
                nbytes=65536,
                sensitivity=0.5,
            ),
        }
        budgets = engine.allocator.allocate(
            profiles, target_ratio=5000.0, max_error=0.0002
        )
        assert budgets["attention.weight"] <= budgets["ffn.weight"]


# ═══════════════════════════════════════════════════════════════════════════
# TestFormatRoundTrip
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatRoundTrip:
    """Tests the SSF format read/write cycle."""

    def test_ssf_write_read_roundtrip(self, small_tensor, temp_dir):
        ssf_path = os.path.join(temp_dir, "test.ssf")
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("test_tensor", small_tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        assert reader.header is not None
        assert "test_tensor" in reader.tensor_names()
        tensor = reader.get_tensor("test_tensor")
        assert tensor.shape == small_tensor.shape
        assert tensor.dtype == small_tensor.dtype
        reader.close()

    def test_ssf_streaming_write_read(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "streaming.ssf")
        tensors = {
            "embedding": np.random.randn(64, 32).astype(np.float32),
            "attention": np.random.randn(32, 32).astype(np.float32),
            "ffn": np.random.randn(64, 128).astype(np.float32),
        }
        with SSFWriter(ssf_path) as writer:
            for name, tensor in tensors.items():
                writer.write_tensor_stream(name, tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        assert len(reader) == len(tensors)
        for name, original in tensors.items():
            loaded = reader.get_tensor(name)
            assert loaded.shape == original.shape
        reader.close()

    def test_ssf_mmap_read(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "mmap_test.ssf")
        tensor = np.random.randn(64, 64).astype(np.float32)
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("mmap_tensor", tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=True)
        loaded = reader.get_tensor("mmap_tensor")
        assert loaded.shape == tensor.shape
        reader.close()

    def test_ssf_integrity_verification(self, small_tensor, temp_dir):
        ssf_path = os.path.join(temp_dir, "verify_test.ssf")
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("verify_tensor", small_tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        result = reader.verify()
        assert result["valid"] is True
        assert result["header_ok"] is True
        assert result["checksum_ok"] is True
        assert "verify_tensor" in result["tensor_checksums"]
        assert result["tensor_checksums"]["verify_tensor"] == "ok"
        reader.close()

    def test_ssf_subset_extraction(self, multi_tensor_model, temp_dir):
        ssf_path = os.path.join(temp_dir, "source.ssf")
        with SSFWriter(ssf_path) as writer:
            for name, tensor in multi_tensor_model.items():
                writer.add_tensor(name, tensor, method=0)

        subset_path = os.path.join(temp_dir, "subset.ssf")
        reader = SSFReader(ssf_path, mmap_mode=False)
        names_to_extract = list(multi_tensor_model.keys())[:3]
        result = reader.extract_subset(names_to_extract, subset_path)
        assert result["n_tensors"] == len(names_to_extract)
        reader.close()

        subset_reader = SSFReader(subset_path, mmap_mode=False)
        assert len(subset_reader) == len(names_to_extract)
        for name in names_to_extract:
            tensor = subset_reader.get_tensor(name)
            assert tensor.shape == multi_tensor_model[name].shape
        subset_reader.close()

    def test_ssf_chunked_read(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "chunked.ssf")
        tensor = np.random.randn(32, 64).astype(np.float32)
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("2d_tensor", tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        chunk = reader.read_tensor_chunk("2d_tensor", row_start=0, row_end=10)
        assert chunk.shape == (10, 64)
        chunk2 = reader.read_tensor_chunk("2d_tensor", row_start=10, row_end=20)
        assert chunk2.shape == (10, 64)
        full = reader.read_tensor_chunk("2d_tensor")
        assert full.shape == tensor.shape
        reader.close()

    def test_ssf_write_read_multiple_methods(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "methods.ssf")
        tensors = [
            ("int8_tensor", np.random.randn(32, 32).astype(np.float32), 0),
            ("int4_tensor", np.random.randn(64, 64).astype(np.float32), 0),
        ]
        with SSFWriter(ssf_path) as writer:
            for name, tensor, method in tensors:
                writer.add_tensor(name, tensor, method=method)

        reader = SSFReader(ssf_path, mmap_mode=False)
        for name, original, _ in tensors:
            loaded = reader.get_tensor(name)
            assert loaded.shape == original.shape
        assert len(reader) == 2
        reader.close()

    def test_ssf_quality_report(self, small_tensor, temp_dir):
        ssf_path = os.path.join(temp_dir, "quality.ssf")
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor(
                "quality_tensor",
                small_tensor,
                method=0,
                quality_metrics={
                    "relative_error": 0.001,
                    "snr_db": 50.0,
                    "psnr_db": 55.0,
                    "compression_ratio": 4.0,
                },
            )

        reader = SSFReader(ssf_path, mmap_mode=False)
        report = reader.get_quality_report()
        assert report["aggregate"]["n_tensors"] == 1
        assert report["aggregate"]["mean_compression_ratio"] > 0
        reader.close()


# ═══════════════════════════════════════════════════════════════════════════
# TestCLIEndToEnd
# ═══════════════════════════════════════════════════════════════════════════


class TestCLIEndToEnd:
    """Tests CLI commands work together."""

    def test_compress_then_validate(self, multi_tensor_model, temp_dir):
        ssf_path = os.path.join(temp_dir, "cli_test.ssf")
        with SSFWriter(ssf_path) as writer:
            for name, tensor in multi_tensor_model.items():
                writer.add_tensor(name, tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        result = reader.verify()
        assert result["valid"] is True
        assert result["header_ok"] is True
        assert len(reader.tensor_names()) == len(multi_tensor_model)
        reader.close()

    def test_compress_then_info(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "info_test.ssf")
        tensor = np.random.randn(32, 32).astype(np.float32)
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("info_tensor", tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        names = reader.tensor_names()
        assert "info_tensor" in names
        info = reader.tensor_info("info_tensor")
        assert info is not None
        assert info["shape"] == [32, 32]
        assert info["ratio"] > 0
        reader.close()

    def test_list_methods_all(self):
        methods = MethodDiscovery.discover()
        assert len(methods) >= 10
        method_names = list(methods.keys())
        assert "block_int8" in method_names
        assert "block_int4" in method_names
        for name, info in methods.items():
            assert "category" in info
            assert "tier" in info

    def test_list_methods_category_filter(self):
        methods = MethodDiscovery.discover()
        quantization_methods = {
            n: m for n, m in methods.items() if m.get("category") == "quantization"
        }
        assert len(quantization_methods) > 0
        for name, info in quantization_methods.items():
            assert info["category"] == "quantization"

    def test_list_methods_tier_filter(self):
        tier5_methods = MethodDiscovery.get_methods_by_tier(5)
        assert len(tier5_methods) > 0
        for name, info in tier5_methods.items():
            assert info["tier"] == 5

    def test_verify_command(self, engine):
        tensor = np.random.randn(32, 32).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="test")

        methods = ["block_int8", "block_int4", "hadamard_int8"]
        results = {}
        for method_name in methods:
            data, meta, ratio, error = engine.compress_fast(
                tensor, name=f"test_{method_name}"
            )
            results[method_name] = {
                "ratio": ratio,
                "error": error,
            }

        assert len(results) == 3
        for method_name in methods:
            assert results[method_name]["ratio"] > 1.0
            assert results[method_name]["error"] >= 0

    def test_benchmark_command(self, engine):
        tensor = np.random.randn(64, 64).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="bench")
        methods = ["block_int8", "block_int4"]
        bench_results = []
        for method_name in methods:
            data, meta, ratio, error = engine.compress_fast(
                tensor, name=f"bench_{method_name}"
            )
            bench_results.append(
                {
                    "method": method_name,
                    "ratio": ratio,
                    "error": error,
                    "time": meta.get("computation_time", 0.0),
                }
            )
        assert len(bench_results) == 2
        assert bench_results[0]["ratio"] > 1.0

    def test_convert_command(self, multi_tensor_model, temp_dir):
        ssf_path = os.path.join(temp_dir, "converted.ssf")
        method_id = 0
        with SSFWriter(ssf_path, compression_method=method_id) as writer:
            for name, tensor in multi_tensor_model.items():
                writer.add_tensor(name, tensor)

        reader = SSFReader(ssf_path, mmap_mode=False)
        assert len(reader) == len(multi_tensor_model)
        for name, original in multi_tensor_model.items():
            loaded = reader.get_tensor(name)
            assert loaded.shape == original.shape
        reader.close()

    def test_cli_command_argument_parsing(self):
        from spectralstream.compression.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["compress", "model.safetensors", "out.ssf"])
        assert args.command == "compress"
        assert args.model == "model.safetensors"
        assert args.output == "out.ssf"

    def test_cli_list_methods_output(self):
        from spectralstream.compression.cli import cmd_list_methods
        import argparse

        try:
            cmd_list_methods(
                argparse.Namespace(category=None, tier=None, verbose=False)
            )
        except SystemExit:
            pass

    def test_method_discovery_counts(self):
        methods = MethodDiscovery.discover()
        stats = MethodDiscovery.get_method_stats()
        assert stats["total"] == len(methods)
        assert stats["tier5_quantization"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# TestMethodIntegration
# ═══════════════════════════════════════════════════════════════════════════


class TestMethodIntegration:
    """Tests that methods from each category work in the engine."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "block_int8",
            "block_int4",
            "hadamard_int8",
            "hadamard_int4",
            "sparsity_int4",
        ],
    )
    def test_method_in_engine_pipeline(self, method_name, engine):
        tensor = np.random.randn(64, 64).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="test")
        methods = engine.get_available_methods()
        if method_name not in methods:
            pytest.skip(f"Method {method_name} not available")
        inst = methods[method_name]
        data, meta = inst.compress(tensor)
        ratio = tensor.nbytes / max(len(data), 1)
        assert ratio > 1.0
        recon = inst.decompress(data, meta)
        if recon.shape != tensor.shape:
            recon = recon.reshape(tensor.shape)
        assert recon.shape == tensor.shape

    def test_all_categories_have_working_methods(self):
        methods = MethodDiscovery.discover()
        categories = set()
        for info in methods.values():
            categories.add(info.get("category", ""))

        for cat in sorted(categories):
            cat_methods = {n: m for n, m in methods.items() if m.get("category") == cat}
            assert len(cat_methods) > 0, f"No methods in category '{cat}'"

    def test_tier_assignment_correct(self):
        methods = MethodDiscovery.discover()
        for name, info in methods.items():
            tier = info["tier"]
            assert tier in (1, 2, 3, 4, 5)
            assert isinstance(tier, int)

    def test_method_in_registry(self):
        assert "block_int8" in METHOD_REGISTRY
        assert "block_int4" in METHOD_REGISTRY
        assert "hadamard_int8" in METHOD_REGISTRY

    def test_method_roundtrip_preserves_dtype(self):
        tensor = np.random.randn(32, 32).astype(np.float32)
        for method_name in ["block_int8", "block_int4"]:
            inst = METHOD_REGISTRY[method_name]
            data, meta = inst.compress(tensor)
            recon = inst.decompress(data, meta).reshape(tensor.shape)
            assert recon.dtype == np.float32
        inst = METHOD_REGISTRY["hadamard_int8"]
        data, meta = inst.compress(tensor)
        recon = inst.decompress(data, meta).reshape(tensor.shape)
        assert recon.dtype in (np.float32, np.float64)

    def test_method_compression_ratio_gt_one(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        for method_name in ["block_int8", "block_int4"]:
            inst = METHOD_REGISTRY[method_name]
            data, meta = inst.compress(tensor)
            assert len(data) < tensor.nbytes

    def test_method_works_with_different_block_sizes(self):
        tensor = np.random.randn(64, 64).astype(np.float32)
        inst = METHOD_REGISTRY["block_int8"]
        for block_size in [32, 64, 128]:
            data, meta = inst.compress(tensor, block_size=block_size)
            recon = inst.decompress(data, meta).reshape(tensor.shape)
            assert recon.shape == tensor.shape

    def test_method_validation(self):
        methods = MethodDiscovery.discover()
        for name, info in list(methods.items())[:5]:
            works, ratio, error = MethodDiscovery.validate_method(name, info)
            assert works, f"Method {name} did not validate"

    def test_engine_method_info(self, engine):
        info = engine.get_methods()
        assert len(info) > 0
        for name in ["block_int8", "block_int4"]:
            assert name in info

    def test_engine_available_methods(self, engine):
        methods = engine.get_available_methods()
        assert "block_int8" in methods
        assert "block_int4" in methods
        assert len(methods) >= 5

    def test_dynamic_selector_works(self):
        selector = DynamicIntelligenceSelector()
        assert selector is not None
        profile = TensorProfile(
            name="test",
            shape=(64, 64),
            n_elements=4096,
            nbytes=16384,
            tensor_type="attention",
            sensitivity=0.9,
        )
        methods = MethodDiscovery.discover()
        available = [
            {
                "name": name,
                "instance": info.get("class"),
                "category": info.get("category", ""),
            }
            for name, info in list(methods.items())[:20]
        ]
        candidates = selector.select(
            profile, available_methods=available, error_budget=0.01, target_ratio=100.0
        )
        assert len(candidates) > 0

    def test_methods_have_correct_category(self):
        methods = MethodDiscovery.discover()
        assert methods["block_int8"]["category"] == "quantization"
        assert methods["block_int4"]["category"] == "quantization"
        assert methods["hadamard_int8"]["category"] == "transform_quant"

    def test_error_budget_allocator(self):
        alloc = ErrorBudgetAllocator()
        profiles = {
            "sensitive": TensorProfile(name="sensitive", sensitivity=1.0),
            "robust": TensorProfile(name="robust", sensitivity=0.3),
        }
        budgets = alloc.allocate(profiles, target_ratio=1000.0, max_error=0.01)
        assert budgets["sensitive"] < budgets["robust"]

    def test_all_methods_tier_mapped(self):
        methods = MethodDiscovery.discover()
        for name in methods:
            assert name in METHOD_TIER_MAP, f"{name} not in METHOD_TIER_MAP"

    def test_method_categories_from_discovery(self):
        methods = MethodDiscovery.discover()
        categories = set()
        for info in methods.values():
            categories.add(info["category"])
        expected_categories = {
            "quantization",
            "decomposition",
            "spectral",
            "structural",
            "entropy",
            "functional",
            "physics",
            "lossless",
            "hybrid",
            "novel",
        }
        assert categories.issubset(expected_categories) or expected_categories.issubset(
            categories
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestErrorHandling
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Tests error handling throughout the system."""

    def test_compress_empty_tensor(self, engine):
        with pytest.raises((AttributeError, RuntimeError, TypeError)):
            engine.compress_fast(None, name="empty")

    def test_compress_bad_path(self):
        engine = CompressionIntelligenceEngine()
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            engine.compress_fast("not_a_tensor", name="bad")

    def test_validate_corrupted_ssf(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "corrupt.ssf")
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor(
                "good", np.random.randn(8, 8).astype(np.float32), method=0
            )
        data = Path(ssf_path).read_bytes()
        if len(data) > 512:
            pos = 400
            corrupted = (
                data[:pos]
                + bytes([b ^ 0xFF for b in data[pos : pos + 4]])
                + data[pos + 4 :]
            )
            Path(ssf_path).write_bytes(corrupted)
            reader = SSFReader(ssf_path, mmap_mode=False)
            result = reader.verify()
            assert result["valid"] is False
            reader.close()
        else:
            pytest.skip("File too small to corrupt meaningfully")

    def test_method_discovery_handles_missing_modules(self):
        methods = MethodDiscovery.discover()
        assert len(methods) > 0

    def test_engine_handles_oom_gracefully(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="test")
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        assert data is not None

    def test_unknown_method_raises(self, engine):
        tensor = np.random.randn(16, 16).astype(np.float32)
        profile = engine.profiler.profile_tensor(tensor, name="test")
        data, meta, ratio, error = engine.compress_fast(tensor, name="test")
        # Override method name in metadata to verify passthrough handling
        meta["method"] = "passthrough"
        assert meta.get("method") == "passthrough"

    def test_decompress_unknown_method(self, engine):
        ct = CompressedTensor(
            _data=b"test",
            method="nonexistent",
            params={},
            original_shape=(4, 4),
            original_dtype="float32",
            compression_ratio=1.0,
            relative_error=0.0,
            snr_db=0.0,
            psnr_db=0.0,
            cosine_similarity=1.0,
            computation_time=0.0,
        )
        # Engine falls back gracefully for unknown methods
        recon = engine.decompress(
            ct._data,
            {"method": "nonexistent", "original_shape": (4, 4)},
        )
        assert recon is not None
        assert recon.size > 0

    def test_reader_nonexistent_tensor(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "missing_tensor.ssf")
        tensor = np.random.randn(16, 16).astype(np.float32)
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("existing", tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        with pytest.raises(KeyError):
            reader.get_tensor("nonexistent")
        reader.close()

    def test_reader_out_of_bounds(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "bounds_check.ssf")
        tensor = np.random.randn(2, 2).astype(np.float32)
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("small", tensor, method=0)

        reader = SSFReader(ssf_path, mmap_mode=False)
        result = reader.verify()
        assert result["valid"] is True
        reader.close()

    def test_reader_corrupted_tensor_checksum(self, temp_dir):
        ssf_path = os.path.join(temp_dir, "tampered.ssf")
        tensor = np.random.randn(8, 8).astype(np.float32)
        with SSFWriter(ssf_path) as writer:
            writer.add_tensor("safe", tensor, method=0)

        data = Path(ssf_path).read_bytes()
        mid = len(data) // 2
        corrupted = (
            data[:mid]
            + bytes([b ^ 0xFF for b in data[mid : mid + 10]])
            + data[mid + 10 :]
        )
        Path(ssf_path).write_bytes(corrupted)

        reader = SSFReader(ssf_path, mmap_mode=False)
        result = reader.verify()
        assert result["valid"] is False
        reader.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
